#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#define TK_STANDALONE
#include "../TK_quantisation/nvfp4_v5/core.cuh"
#include "../TK_quantisation/nvfp4_v5/quantize_transpose_tuned.cuh"
#include "../TK_quantisation/nvfp4_v5/persistent_quantize.cuh"

namespace {

constexpr int TILE = 128;
constexpr int FP4_BLOCK = 16;
constexpr int THREADS = 256;
constexpr int ROW_THREADS = 256;
constexpr float NVFP4_GLOBAL_SCALE_DENOM = 2688.0f;  // fp8_e4m3 max 448 * fp4_e2m1 max 6

using CceRNGState = transformer_engine::curanddx::detail::philox4x32_native_state<10>;

constexpr unsigned long long CCE_SR_INVOCATION_STRIDE = 1ull << 40;
constexpr unsigned long long CCE_SR_RANK_STRIDE = 1ull << 56;
__device__ unsigned long long cce_sr_invocation_offset = 0;

__global__ void prepare_cce_advancing_rng_state_kernel(
    unsigned long long* rng_state,
    unsigned long long rng_seed,
    unsigned long long rng_subsequence_base) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long offset = atomicAdd(
            &cce_sr_invocation_offset, CCE_SR_INVOCATION_STRIDE);
        rng_state[0] = rng_seed;
        rng_state[1] = rng_subsequence_base + offset;
    }
}

torch::Tensor make_cce_advancing_rng_state(
    const torch::Tensor& input,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream) {
    auto rng_state = torch::empty(
        {2}, torch::dtype(torch::kInt64).device(input.device()));
    prepare_cce_advancing_rng_state_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
        static_cast<unsigned long long>(rng_seed),
        static_cast<unsigned long long>(rng_subsequence_base));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rng_state;
}

inline bool env_flag(const char* name, bool default_value = false) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return !(value[0] == '\0' || value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

inline uint64_t env_u64(const char* name, uint64_t default_value = 0) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
}

inline float env_float(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return std::strtof(value, nullptr);
}

inline float nvfp4_logit_temperature() {
    const char* value = std::getenv("FP4_CCE_V4_LOGIT_TEMPERATURE");
    if (value == nullptr || value[0] == '\0') return 1.0f;
    const float temperature = std::strtof(value, nullptr);
    TORCH_CHECK(
        std::isfinite(temperature) && temperature >= 0.5f && temperature <= 2.0f,
        "FP4_CCE_V4_LOGIT_TEMPERATURE must be in [0.5, 2]");
    return temperature;
}

struct Nvfp4CceExtras {
    bool data_sr;
    bool row_data_sr;
    bool col_data_sr;
    bool col_zero_sr;
    bool scale_sr;
    bool with_rht;
    bool random_sign_mask;
    uint64_t rng_seed;
    uint64_t rng_subsequence_base;
};

inline Nvfp4CceExtras nvfp4_cce_extras_from_env() {
    Nvfp4CceExtras extras;
    extras.data_sr =
        env_flag("FP4_CCE_V4_NVFP4_DATA_SR", false) ||
        env_flag("FP4_CCE_V4_NVFP4_USE_STOCHASTIC_ROUNDING", false);
    extras.row_data_sr = extras.data_sr ||
        env_flag("FP4_CCE_V4_NVFP4_G_ROW_DATA_SR", false);
    extras.col_data_sr = extras.data_sr ||
        env_flag("FP4_CCE_V4_NVFP4_G_COL_DATA_SR", false);
    extras.col_zero_sr = !extras.col_data_sr &&
        env_flag("FP4_CCE_V4_NVFP4_G_COL_ZERO_SR", false);
    extras.scale_sr =
        env_flag("FP4_CCE_V4_NVFP4_SCALE_SR", false) ||
        env_flag("FP4_CCE_V4_NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", false);
    extras.with_rht =
        env_flag("FP4_CCE_V4_NVFP4_RHT", false) ||
        env_flag("FP4_CCE_V4_NVFP4_USE_RHT", false);
    // Keep the default cancellation contract aligned with the trainer path:
    // pure normalized H16 unless random signs are explicitly enabled.
    extras.random_sign_mask = extras.with_rht && env_flag("FP4_CCE_V4_NVFP4_RHT_RANDOM_SIGNS", false);
    TORCH_CHECK(
        !extras.with_rht,
        "FP4_CCE_V4_NVFP4_USE_RHT is not safe yet: staged P/G RHT only cancels "
        "if the paired CCE X/W column quantizers apply the matching H16 transform "
        "on the GEMM reduction dimension. Enable NVFP4 data/scale SR for CCE, "
        "or port the paired X/W RHT producer first.");
    extras.rng_seed = env_u64("FP4_CCE_V4_NVFP4_RNG_SEED", 0);
    extras.rng_subsequence_base =
        env_u64("FP4_CCE_V4_NVFP4_RNG_SUBSEQUENCE_BASE", 0) +
        env_u64("RANK", 0) * CCE_SR_RANK_STRIDE;
    return extras;
}

inline torch::Tensor maybe_make_cce_advancing_rng_state(
    const torch::Tensor& input,
    const Nvfp4CceExtras& extras,
    cudaStream_t stream) {
    return (extras.row_data_sr || extras.col_data_sr || extras.col_zero_sr ||
            extras.scale_sr || extras.random_sign_mask)
        ? make_cce_advancing_rng_state(
              input, extras.rng_seed, extras.rng_subsequence_base, stream)
        : torch::Tensor();
}

static void create_tma_2d(
    CUtensorMap &map, void *ptr,
    uint64_t globalY, uint64_t globalX,
    uint32_t shmemY, uint32_t shmemX,
    uint64_t strideX, size_t type_num_bits,
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
) {
    typedef CUresult (*cuTensorMapEncodeTiled_t)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (!fn) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        TORCH_CHECK(handle != nullptr, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dataType;
    uint64_t globalDims[2] = {globalX, globalY};
    uint32_t boxDims[2] = {shmemX, shmemY};
    uint64_t globalStrides[1] = {(strideX * type_num_bits) / 8};
    uint32_t elementStrides[2] = {1, 1};

    if (type_num_bits == 16) dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (type_num_bits == 8) dataType = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (type_num_bits == 4) dataType = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported type_num_bits: ", type_num_bits);

    auto result = fn(&map, dataType, 2, ptr,
        globalDims, globalStrides, boxDims, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

inline int select_staged_row_threads(const char* env_value) {
    if (env_value != nullptr) {
        const int parsed = std::atoi(env_value);
        if (parsed == 128 || parsed == 256 || parsed == 512 || parsed == 1024) {
            return parsed;
        }
    }
    return ROW_THREADS;
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float warp_red[THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) warp_red[warp] = v;
    __syncthreads();
    v = (warp == 0 && lane < (THREADS / 32)) ? warp_red[lane] : 0.0f;
    if (warp == 0) v = warp_reduce_max(v);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_red[THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_red[warp] = v;
    __syncthreads();
    v = (warp == 0 && lane < (THREADS / 32)) ? warp_red[lane] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ void atomic_max_float_positive(float* addr, float value) {
    if (value <= 0.0f) return;
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value));
}

__device__ __forceinline__ float global_encode_scale(
    float global_amax,
    float global_scale_denom) {
    if (global_amax == 0.0f) return 1.0f;
    const float scale = global_scale_denom / global_amax;
    return (scale == 0.0f) ? 1.0f : fminf(scale, FLT_MAX);
}

__device__ __forceinline__ uint32_t cce_next_rbits(
    CceRNGState& rng,
    uint4& random_uint4,
    int& rnd_idx) {
    if (rnd_idx == 4) {
        rnd_idx = 0;
        random_uint4 = rng.generate4();
    }
    const uint32_t* rbits_arr = reinterpret_cast<uint32_t*>(&random_uint4);
    return rbits_arr[rnd_idx++];
}

template <bool WITH_RANDOM_SIGN_MASK>
__device__ __forceinline__ uint32_t cce_make_rht_sign_bits(
    CceRNGState& rng,
    uint4& random_uint4,
    int& rnd_idx) {
    if constexpr (!WITH_RANDOM_SIGN_MASK) {
        return 0xffffffffu;
    }
    return cce_next_rbits(rng, random_uint4, rnd_idx);
}

__device__ __forceinline__ void cce_fwht16_unnormalized(float (&vals)[FP4_BLOCK]) {
    #pragma unroll
    for (int step = 1; step < FP4_BLOCK; step <<= 1) {
        #pragma unroll
        for (int base = 0; base < FP4_BLOCK; base += 2 * step) {
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
__device__ __forceinline__ void cce_apply_rht16(float (&vals)[FP4_BLOCK], uint32_t sign_bits) {
    #pragma unroll
    for (int i = 0; i < FP4_BLOCK; ++i) {
        if constexpr (WITH_RANDOM_SIGN_MASK) {
            vals[i] *= ((sign_bits >> i) & 1u) ? 1.0f : -1.0f;
        }
    }
    cce_fwht16_unnormalized(vals);
    constexpr float kNorm16 = 0.25f;
    #pragma unroll
    for (int i = 0; i < FP4_BLOCK; ++i) {
        vals[i] *= kNorm16;
    }
}

__device__ __forceinline__ __nv_fp8_e4m3 cce_float_to_e4m3_sr(float val, uint32_t rbits) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t packed = 0;
    asm volatile(
        "{\n\t"
        "cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %1, %1, %1}, %2;\n\t"
        "}\n\t"
        : "=r"(packed)
        : "f"(val), "r"(rbits));
    __nv_fp8_e4m3 out;
    *reinterpret_cast<unsigned char*>(&out) = static_cast<unsigned char>(packed & 0xffu);
    return out;
#else
    return static_cast<__nv_fp8_e4m3>(val);
#endif
}

__device__ __forceinline__ uint8_t fp8_byte(__nv_fp8_e4m3 val) {
    return *reinterpret_cast<uint8_t*>(&val);
}

template <bool SCALE_SR>
__device__ __forceinline__ float nvfp4_quant_coeff_opt(
    float block_amax,
    float s_enc,
    CceRNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    uint8_t* scale_byte,
    float scale_fold) {
    __nv_fp8_e4m3 mult_fp8;
    if (block_amax <= 1.0e-9f) {
        mult_fp8 = static_cast<__nv_fp8_e4m3>(448.0f);
    } else {
        const float mult = fminf(6.0f / (block_amax * s_enc), FLT_MAX);
        mult_fp8 = SCALE_SR
            ? cce_float_to_e4m3_sr(mult, cce_next_rbits(rng, random_uint4, rnd_idx))
            : static_cast<__nv_fp8_e4m3>(mult);
    }
    const float mult_f = static_cast<float>(mult_fp8);
    const float inv_mult = (mult_f > 0.0f) ? (1.0f / mult_f) : 448.0f;
    const float stored_scale = inv_mult * scale_fold;
    const __nv_fp8_e4m3 scale_fp8 = SCALE_SR
        ? cce_float_to_e4m3_sr(stored_scale, cce_next_rbits(rng, random_uint4, rnd_idx))
        : static_cast<__nv_fp8_e4m3>(stored_scale);
    *scale_byte = fp8_byte(scale_fp8);
    return mult_f * s_enc;
}

__device__ __forceinline__ float nvfp4_quant_coeff(float block_amax, float s_enc, uint8_t* scale_byte) {
    CceRNGState rng;
    uint4 random_uint4{};
    int rnd_idx = 4;
    return nvfp4_quant_coeff_opt<false>(
        block_amax, s_enc, rng, random_uint4, rnd_idx, scale_byte, 1.0f);
}

__device__ __forceinline__ float nvfp4_quant_coeff_direct_scale(
    float block_amax,
    float s_enc,
    uint8_t* scale_byte) {
    constexpr float MIN_E4M3_SCALE = 1.0f / 512.0f;
    const float raw_scale = fmaxf(
        block_amax * (s_enc / 6.0f), MIN_E4M3_SCALE);
    const __nv_fp8_e4m3 scale_fp8 = static_cast<__nv_fp8_e4m3>(raw_scale);
    const float scale = static_cast<float>(scale_fp8);
    *scale_byte = fp8_byte(scale_fp8);
    return s_enc / scale;
}

template <bool DATA_SR>
__device__ __forceinline__ void pack_nvfp4_block(
    const float (&vals)[FP4_BLOCK],
    float coeff,
    CceRNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    uint8_t (&packed_vals)[8]) {
    if constexpr (DATA_SR) {
        #pragma unroll
        for (int k = 0; k < FP4_BLOCK; k += 4) {
            const float2 in01 = {vals[k + 0], vals[k + 1]};
            const float2 in23 = {vals[k + 2], vals[k + 3]};
            const float2 scale = {coeff, coeff};
            const auto packed4 = transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                in01, in23, scale, cce_next_rbits(rng, random_uint4, rnd_idx));
            const uint16_t raw = *reinterpret_cast<const uint16_t*>(&packed4);
            packed_vals[k >> 1] = static_cast<uint8_t>(raw & 0xffu);
            packed_vals[(k >> 1) + 1] = static_cast<uint8_t>((raw >> 8) & 0xffu);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < FP4_BLOCK; k += 2) {
            const float2 scaled = {vals[k] * coeff, vals[k + 1] * coeff};
            packed_vals[k >> 1] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
        }
    }
}

__device__ __forceinline__ uint16_t select_stochastic_rounded_zeros(
    uint16_t deterministic,
    uint16_t stochastic) {
    uint16_t zero_mask = 0;
    #pragma unroll
    for (int nibble = 0; nibble < 4; ++nibble) {
        const int shift = nibble * 4;
        if (((deterministic >> shift) & 0x7u) == 0) {
            zero_mask |= static_cast<uint16_t>(0xfu << shift);
        }
    }
    return static_cast<uint16_t>(
        (deterministic & ~zero_mask) | (stochastic & zero_mask));
}

__device__ __forceinline__ void pack_nvfp4_block_zero_sr(
    const float (&vals)[FP4_BLOCK],
    float coeff,
    CceRNGState& rng,
    uint4& random_uint4,
    int& rnd_idx,
    uint8_t (&packed_vals)[8]) {
    #pragma unroll
    for (int k = 0; k < FP4_BLOCK; k += 4) {
        const float2 in01 = {vals[k + 0], vals[k + 1]};
        const float2 in23 = {vals[k + 2], vals[k + 3]};
        const float2 scale = {coeff, coeff};
        const auto deterministic4 =
            transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<false>(
                in01, in23, scale, 0);
        const auto stochastic4 =
            transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                in01,
                in23,
                scale,
                cce_next_rbits(rng, random_uint4, rnd_idx));
        const uint16_t deterministic =
            *reinterpret_cast<const uint16_t*>(&deterministic4);
        const uint16_t stochastic =
            *reinterpret_cast<const uint16_t*>(&stochastic4);
        const uint16_t selected = select_stochastic_rounded_zeros(
            deterministic, stochastic);
        packed_vals[k >> 1] = static_cast<uint8_t>(selected & 0xffu);
        packed_vals[(k >> 1) + 1] =
            static_cast<uint8_t>((selected >> 8) & 0xffu);
    }
}

__device__ __forceinline__ void pack_nvfp4_log_affine_block(
    const float (&log_values)[FP4_BLOCK],
    float coeff,
    float affine_a,
    float affine_b,
    uint8_t (&packed_vals)[8]) {
    constexpr float LOG2E = 1.4426950408889634f;
    const float log2_coeff = __log2f(coeff);
    #pragma unroll
    for (int k = 0; k < FP4_BLOCK; k += 2) {
        const float x0 = fmaf(log_values[k], LOG2E, log2_coeff);
        const float x1 = fmaf(log_values[k + 1], LOG2E, log2_coeff);
        const float2 approximated = {
            fmaxf(fmaf(affine_a, x0, affine_b), 0.0f),
            fmaxf(fmaf(affine_a, x1, affine_b), 0.0f),
        };
        packed_vals[k >> 1] = static_cast<uint8_t>(
            __nv_cvt_float2_to_fp4x2(
                approximated, __NV_E2M1, cudaRoundNearest));
    }
}

__device__ __forceinline__ void store_nvfp4_scales(
    uint8_t* __restrict__ scales,
    int tile_row,
    int tile_col128,
    int ntk64,
    int row_in_tile,
    const uint8_t vals[8]) {
    const int j = row_in_tile & 31;
    const int grp = row_in_tile >> 5;
    const int base = (tile_row * ntk64 + tile_col128 * 2) * 512 + j * 16 + grp * 4;
    const uint32_t packed0 =
        static_cast<uint32_t>(vals[0]) |
        (static_cast<uint32_t>(vals[1]) << 8) |
        (static_cast<uint32_t>(vals[2]) << 16) |
        (static_cast<uint32_t>(vals[3]) << 24);
    const uint32_t packed1 =
        static_cast<uint32_t>(vals[4]) |
        (static_cast<uint32_t>(vals[5]) << 8) |
        (static_cast<uint32_t>(vals[6]) << 16) |
        (static_cast<uint32_t>(vals[7]) << 24);
    *reinterpret_cast<uint32_t*>(scales + base) = packed0;
    *reinterpret_cast<uint32_t*>(scales + base + 512) = packed1;
}

__device__ __forceinline__ void store_nvfp4_scale_one(
    uint8_t* __restrict__ scales,
    int tile_row,
    int tile_col128,
    int ntk64,
    int row_in_tile,
    int scale_idx,
    uint8_t val) {
    const int j = row_in_tile & 31;
    const int grp = row_in_tile >> 5;
    const int base = (tile_row * ntk64 + tile_col128 * 2) * 512 + j * 16 + grp * 4;
    scales[base + (scale_idx >> 2) * 512 + (scale_idx & 3)] = val;
}

__device__ __forceinline__ uint8_t load_nvfp4_scale_one(
    const uint8_t* __restrict__ scales,
    int tile_row,
    int tile_col128,
    int ntk64,
    int row_in_tile,
    int scale_idx) {
    const int j = row_in_tile & 31;
    const int grp = row_in_tile >> 5;
    const int base =
        (tile_row * ntk64 + tile_col128 * 2) * 512 + j * 16 + grp * 4;
    return scales[base + (scale_idx >> 2) * 512 + (scale_idx & 3)];
}

__device__ __forceinline__ float e4m3_from_byte(uint8_t raw) {
    __nv_fp8_e4m3 value;
    *reinterpret_cast<uint8_t*>(&value) = raw;
    return static_cast<float>(value);
}

__device__ __forceinline__ float e8m0_from_byte(uint8_t raw) {
    __nv_fp8_e8m0 value;
    value.__x = raw;
    return static_cast<float>(value);
}

__device__ __forceinline__ void store_8_packed_fp4(uint8_t* __restrict__ dst, const uint8_t vals[8]) {
    const uint64_t packed =
        static_cast<uint64_t>(vals[0]) |
        (static_cast<uint64_t>(vals[1]) << 8) |
        (static_cast<uint64_t>(vals[2]) << 16) |
        (static_cast<uint64_t>(vals[3]) << 24) |
        (static_cast<uint64_t>(vals[4]) << 32) |
        (static_cast<uint64_t>(vals[5]) << 40) |
        (static_cast<uint64_t>(vals[6]) << 48) |
        (static_cast<uint64_t>(vals[7]) << 56);
    *reinterpret_cast<uint64_t*>(dst) = packed;
}

template <
    int ROW_THREADS_,
    bool DATA_SR = false,
    bool SCALE_SR = false,
    bool WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ __launch_bounds__(ROW_THREADS_)
void nvfp4_softmax_row_quant_staged_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    __nv_bfloat16* __restrict__ probs,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    float* __restrict__ sg_out,
    float* __restrict__ loss_sum,
    float* __restrict__ count_sum,
    int M,
    int N,
    int vocab_size,
    bool grad_cache,
    float s_enc,
    float sg_value,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* __restrict__ rng_state) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    if constexpr (DATA_SR || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
    }

    constexpr int ROW_WARPS_ = ROW_THREADS_ / 32;
    __shared__ float warp_red[ROW_WARPS_];
    __shared__ float s_max;
    __shared__ float s_inv_sum;
    __shared__ float s_lse;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (row == 0 && tid == 0) {
        sg_out[0] = sg_value;
    }

    const bool is_valid = valid[row];
    const int64_t target = is_valid ? targets[row] : -1;
    CceRNGState rng;
    rng.init(rng_seed, rng_subsequence_base + static_cast<uint64_t>(row) * 4096ull + tid, 0);
    uint4 random_uint4{};
    int rnd_idx = 4;
    float local_max = -INFINITY;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += ROW_THREADS_) {
            local_max = fmaxf(local_max, __bfloat162float(logits[static_cast<int64_t>(row) * N + col]));
        }
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS_) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += ROW_THREADS_) {
            const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
            local_sum += __expf(z - s_max);
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS_) ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            s_inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            s_lse = (sum > 0.0f) ? (s_max + logf(sum)) : 0.0f;
            if (is_valid) {
                const float target_logit = __bfloat162float(logits[static_cast<int64_t>(row) * N + target]);
                atomicAdd(loss_sum, s_lse - target_logit);
                atomicAdd(count_sum, 1.0f);
            }
        }
    }
    __syncthreads();

    const int blocks = N / FP4_BLOCK;
    const int tile_row = row / TILE;
    const int row_in_tile = row - tile_row * TILE;
    const int ntk64 = N / 64;

    for (int block = tid; block < blocks; block += ROW_THREADS_) {
        const int col0 = block * FP4_BLOCK;
        const int tile_col = col0 / TILE;
        const int scale_idx = (col0 / FP4_BLOCK) & 7;
        float vals[FP4_BLOCK];
        float block_amax = 0.0f;

        #pragma unroll
        for (int k = 0; k < FP4_BLOCK; ++k) {
            const int col = col0 + k;
            float p = 0.0f;
            if (is_valid && col < vocab_size) {
                const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
                p = __expf(z - s_max) * s_inv_sum;
                if (grad_cache && col == target) {
                    p -= 1.0f;
                }
            }
            vals[k] = p;
            probs[static_cast<int64_t>(row) * N + col] = __float2bfloat16(p);
        }

        if constexpr (WITH_RHT) {
            const uint32_t sign_bits =
                cce_make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
            cce_apply_rht16<WITH_RANDOM_SIGN_MASK>(vals, sign_bits);
        }
        #pragma unroll
        for (int k = 0; k < FP4_BLOCK; ++k) {
            block_amax = fmaxf(block_amax, fabsf(vals[k]));
        }

        uint8_t scale_byte = 0;
        const float coeff = nvfp4_quant_coeff_opt<SCALE_SR>(
            block_amax, s_enc, rng, random_uint4, rnd_idx, &scale_byte, 1.0f);
        store_nvfp4_scale_one(row_sc, tile_row, tile_col, ntk64, row_in_tile, scale_idx, scale_byte);

        uint8_t packed_vals[8];
        pack_nvfp4_block<DATA_SR>(vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
        store_8_packed_fp4(
            reinterpret_cast<uint8_t*>(row_fp4) + static_cast<int64_t>(row) * (N / 2) + col0 / 2,
            packed_vals);
    }
}

template <
    int ROW_THREADS_,
    bool DATA_SR = false,
    bool SCALE_SR = false,
    bool WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false>
void launch_nvfp4_softmax_row_quant_staged(
    const __nv_bfloat16* logits,
    const int64_t* targets,
    const bool* valid,
    __nv_bfloat16* probs,
    __nv_fp4x2_e2m1* row_fp4,
    uint8_t* row_sc,
    float* sg_out,
    float* loss_sum,
    float* count_sum,
    int M,
    int N,
    int vocab_size,
    bool grad_cache,
    float s_enc,
    float sg_value,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* rng_state,
    cudaStream_t stream) {
    nvfp4_softmax_row_quant_staged_kernel<
        ROW_THREADS_, DATA_SR, SCALE_SR, WITH_RHT, WITH_RANDOM_SIGN_MASK><<<M, ROW_THREADS_, 0, stream>>>(
        logits,
        targets,
        valid,
        probs,
        row_fp4,
        row_sc,
        sg_out,
        loss_sum,
        count_sum,
        M,
        N,
        vocab_size,
        grad_cache,
        s_enc,
        sg_value,
        rng_seed,
        rng_subsequence_base,
        rng_state);
}

template <
    bool DATA_SR = false,
    bool SCALE_SR = false,
    bool WITH_RHT = false,
    bool WITH_RANDOM_SIGN_MASK = false>
__global__ __launch_bounds__(128)
void nvfp4_col_quant_from_probs_kernel(
    const __nv_bfloat16* __restrict__ probs,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    float s_enc,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* __restrict__ rng_state) {
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int c = threadIdx.x;
    if (c >= TILE) return;

    if constexpr (DATA_SR || SCALE_SR || WITH_RANDOM_SIGN_MASK) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
    }

    const int gc = col_base + c;
    const int ntk64 = M / 64;
    uint8_t scales[8];
    CceRNGState rng;
    const uint64_t col_subseq =
        rng_subsequence_base + 0x100000000ull +
        static_cast<uint64_t>(tile_col) * 131072ull +
        static_cast<uint64_t>(tile_row) * 1024ull + c;
    rng.init(rng_seed, col_subseq, 0);
    uint4 random_uint4{};
    int rnd_idx = 4;

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        float vals[FP4_BLOCK];
        float block_amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < FP4_BLOCK; ++j) {
            const int r = b * FP4_BLOCK + j;
            const float p = __bfloat162float(probs[static_cast<int64_t>(row_base + r) * N + gc]);
            vals[j] = p;
        }
        if constexpr (WITH_RHT) {
            const uint32_t sign_bits =
                cce_make_rht_sign_bits<WITH_RANDOM_SIGN_MASK>(rng, random_uint4, rnd_idx);
            cce_apply_rht16<WITH_RANDOM_SIGN_MASK>(vals, sign_bits);
        }
        #pragma unroll
        for (int j = 0; j < FP4_BLOCK; ++j) {
            block_amax = fmaxf(block_amax, fabsf(vals[j]));
        }
        uint8_t scale_byte = 0;
        const float coeff = nvfp4_quant_coeff_opt<SCALE_SR>(
            block_amax, s_enc, rng, random_uint4, rnd_idx, &scale_byte, 1.0f);
        scales[b] = scale_byte;

        uint8_t packed_vals[8];
        pack_nvfp4_block<DATA_SR>(vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
        store_8_packed_fp4(
            reinterpret_cast<uint8_t*>(col_fp4) + static_cast<int64_t>(gc) * (M / 2) + (row_base + b * FP4_BLOCK) / 2,
            packed_vals);
    }
    store_nvfp4_scales(col_sc, tile_col, tile_row, ntk64, c, scales);
}

template <int ROW_THREADS_>
void launch_nvfp4_softmax_row_quant_staged_selected(
    const __nv_bfloat16* logits,
    const int64_t* targets,
    const bool* valid,
    __nv_bfloat16* probs,
    __nv_fp4x2_e2m1* row_fp4,
    uint8_t* row_sc,
    float* sg_out,
    float* loss_sum,
    float* count_sum,
    int M,
    int N,
    int vocab_size,
    bool grad_cache,
    float s_enc,
    float sg_value,
    bool data_sr,
    bool scale_sr,
    bool with_rht,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* rng_state,
    cudaStream_t stream) {
#define LAUNCH_STAGED_ROW(DATA, SCALE, RHT, SIGN) \
    launch_nvfp4_softmax_row_quant_staged<ROW_THREADS_, DATA, SCALE, RHT, SIGN>( \
        logits, targets, valid, probs, row_fp4, row_sc, sg_out, loss_sum, count_sum, \
        M, N, vocab_size, grad_cache, s_enc, sg_value, rng_seed, rng_subsequence_base, \
        rng_state, stream)
    if (with_rht) {
        if (with_random_sign_mask) {
            if (data_sr && scale_sr) { LAUNCH_STAGED_ROW(true, true, true, true); }
            else if (data_sr) { LAUNCH_STAGED_ROW(true, false, true, true); }
            else if (scale_sr) { LAUNCH_STAGED_ROW(false, true, true, true); }
            else { LAUNCH_STAGED_ROW(false, false, true, true); }
        } else {
            if (data_sr && scale_sr) { LAUNCH_STAGED_ROW(true, true, true, false); }
            else if (data_sr) { LAUNCH_STAGED_ROW(true, false, true, false); }
            else if (scale_sr) { LAUNCH_STAGED_ROW(false, true, true, false); }
            else { LAUNCH_STAGED_ROW(false, false, true, false); }
        }
    } else {
        if (data_sr && scale_sr) { LAUNCH_STAGED_ROW(true, true, false, false); }
        else if (data_sr) { LAUNCH_STAGED_ROW(true, false, false, false); }
        else if (scale_sr) { LAUNCH_STAGED_ROW(false, true, false, false); }
        else { LAUNCH_STAGED_ROW(false, false, false, false); }
    }
#undef LAUNCH_STAGED_ROW
}

void launch_nvfp4_col_quant_from_probs_selected(
    const __nv_bfloat16* probs,
    __nv_fp4x2_e2m1* col_fp4,
    uint8_t* col_sc,
    int M,
    int N,
    float s_enc,
    bool data_sr,
    bool scale_sr,
    bool with_rht,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* rng_state,
    dim3 grid,
    cudaStream_t stream) {
#define LAUNCH_COL(DATA, SCALE, RHT, SIGN) \
    nvfp4_col_quant_from_probs_kernel<DATA, SCALE, RHT, SIGN><<<grid, 128, 0, stream>>>( \
        probs, col_fp4, col_sc, M, N, s_enc, rng_seed, rng_subsequence_base, rng_state)
    if (with_rht) {
        if (with_random_sign_mask) {
            if (data_sr && scale_sr) { LAUNCH_COL(true, true, true, true); }
            else if (data_sr) { LAUNCH_COL(true, false, true, true); }
            else if (scale_sr) { LAUNCH_COL(false, true, true, true); }
            else { LAUNCH_COL(false, false, true, true); }
        } else {
            if (data_sr && scale_sr) { LAUNCH_COL(true, true, true, false); }
            else if (data_sr) { LAUNCH_COL(true, false, true, false); }
            else if (scale_sr) { LAUNCH_COL(false, true, true, false); }
            else { LAUNCH_COL(false, false, true, false); }
        }
    } else {
        if (data_sr && scale_sr) { LAUNCH_COL(true, true, false, false); }
        else if (data_sr) { LAUNCH_COL(true, false, false, false); }
        else if (scale_sr) { LAUNCH_COL(false, true, false, false); }
        else { LAUNCH_COL(false, false, false, false); }
    }
#undef LAUNCH_COL
}

template <bool GRAD_CACHE>
__global__ __launch_bounds__(tk_v3::V3_THREADS)
void nvfp4_softmax_quant_row_col_tma_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    float* __restrict__ sg_out,
    int M,
    int N,
    int vocab_size) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using namespace tk_v3;
    using namespace tk_v5;
    using namespace transformer_engine;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType*         sIn_ptr        = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2*     sOut_ptr       = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    fp4e2m1x2*     sOut_tr_ptr    = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes + out_bytes);
    nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes);
    nvfp4_scale_t* sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t*>(dshmem + in_bytes + out_bytes + out_tr_bytes + sc_row_bytes);

    auto& sIn        = *reinterpret_cast<V3_IType3D*>(sIn_ptr);
    auto& sSFrowwise = *reinterpret_cast<V3_ScalesType2D*>(sSFrowwise_ptr);
    auto& sSFcolwise = *reinterpret_cast<V3_ScalesTypeTr2D*>(sSFcolwise_ptr);
    auto& sOut       = *reinterpret_cast<V3_OType2x3D*>(sOut_ptr);
    auto& sOut_tr    = *reinterpret_cast<V3_OType2xt3D*>(sOut_tr_ptr);

    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;
    const int tid = threadIdx.x;

    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        sg_out[0] = 1.0f / NVFP4_GLOBAL_SCALE_DENOM;
    }

    #pragma unroll
    for (int t = 0; t < V3_NUM_TILES; ++t) {
        const int stage_Y = t / V3_TILES_X;
        const int stage_X = t % V3_TILES_X;
        const int stage_offset_Y = stage_Y * V3_TILE_DIM_Y;
        const int stage_offset_X = stage_X * V3_TILE_DIM_X;

        for (int idx = tid; idx < V3_BUFF_IN_ELEMS; idx += V3_THREADS) {
            const int r = idx / V3_BUFF_DIM_X;
            const int c = idx - r * V3_BUFF_DIM_X;
            const int gr = block_offset_Y + stage_offset_Y + r;
            const int gc = block_offset_X + stage_offset_X + c;

            float p = 0.0f;
            if (gr < M && gc < vocab_size && valid[gr]) {
                const float z = __bfloat162float(logits[static_cast<int64_t>(gr) * N + gc]);
                p = __expf(z - lse[gr]);
                if constexpr (GRAD_CACHE) {
                    if (gc == targets[gr]) {
                        p -= 1.0f;
                    }
                }
            }
            sIn[t][r][c] = __float2bfloat16(p);
        }
    }
    __syncthreads();

    quantize_and_store_chunk_v5<true, true>(
        sIn_ptr,
        sOut_ptr,
        sOut_tr_ptr,
        sSFrowwise_ptr,
        sSFcolwise_ptr,
        sOut,
        sOut_tr,
        sSFrowwise,
        sSFcolwise,
        tensor_map_output,
        tensor_map_output_t,
        tmap_scale_row,
        tmap_scale_col,
        NVFP4_GLOBAL_SCALE_DENOM,
        block_offset_Y,
        block_offset_X,
        M,
        N,
        ctaid_X,
        ctaid_Y);
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

__global__ __launch_bounds__(THREADS)
void nvfp4_softmax_amax_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    float* __restrict__ global_amax,
    int M,
    int N,
    int vocab_size,
    float logit_temperature) {
    const int64_t total = static_cast<int64_t>(M) * N;
    float local = 0.0f;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / N);
        const int col = static_cast<int>(idx - static_cast<int64_t>(row) * N);
        if (col < vocab_size && valid[row]) {
            const float z = logit_temperature * __bfloat162float(logits[idx]);
            local = fmaxf(local, __expf(z - lse[row]));
        }
    }
    local = block_reduce_max(local);
    if (threadIdx.x == 0) atomic_max_float_positive(global_amax, local);
}

template <
    bool CONSTANT_SCALE,
    bool GRAD_CACHE,
    bool CHUNK_SCALE,
    bool ROW_DATA_SR = false,
    bool COL_DATA_SR = false,
    bool COL_ZERO_SR = false,
    bool SCALE_SR = false,
    bool LOGSPACE_QUANT = false,
    bool DIRECT_SCALE = false>
__global__ __launch_bounds__(THREADS)
void nvfp4_softmax_quant_row_col_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    const float* __restrict__ global_amax,
    float* __restrict__ row_sg_out,
    float* __restrict__ col_sg_out,
    float* __restrict__ row_sg_outer,
    float* __restrict__ col_sg_outer,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float global_scale_denom,
    float logit_temperature,
    float logspace_affine_a,
    float logspace_affine_b,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* __restrict__ rng_state) {
    // Pad the shared tile to avoid bank conflicts when the column path reads it
    // with a stride of TILE.
    __shared__ __nv_bfloat16 probs[TILE][TILE + 1];

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;

    CceRNGState rng;
    uint4 random_uint4{};
    int rnd_idx = 4;
    if constexpr (ROW_DATA_SR || COL_DATA_SR || COL_ZERO_SR || SCALE_SR) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
        const uint64_t tile_sequence =
            (static_cast<uint64_t>(tile_row) * gridDim.x + tile_col) * 512ull;
        const uint64_t orientation_offset =
            (tid < TILE) ? 0ull : 0x100000000ull;
        rng.init(
            rng_seed,
            rng_subsequence_base + orientation_offset + tile_sequence + tid,
            0);
    }

    for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
        const int r = idx / TILE;
        const int c = idx - r * TILE;
        const int gr = row_base + r;
        const int gc = col_base + c;

        float p = LOGSPACE_QUANT ? -INFINITY : 0.0f;
        if (gr < M && gc < vocab_size && valid[gr]) {
            const float z = logit_temperature *
                __bfloat162float(logits[static_cast<int64_t>(gr) * N + gc]);
            if constexpr (LOGSPACE_QUANT) {
                p = z - lse[gr];
            } else {
                p = __expf(z - lse[gr]);
                if constexpr (GRAD_CACHE) {
                    if (gc == targets[gr]) {
                        p -= 1.0f;
                    }
                }
            }
        }
        probs[r][c] = __float2bfloat16(p);
    }

    float s_enc;
    if constexpr (CHUNK_SCALE) {
        s_enc = global_scale_denom;
        if (tile_row == 0 && tile_col == 0 && tid == 0) {
            const float sg = 1.0f / NVFP4_GLOBAL_SCALE_DENOM;
            row_sg_out[0] = sg;
            col_sg_out[0] = sg;
        }
        __syncthreads();
    } else {
        constexpr bool UNIT_BOUND = CONSTANT_SCALE || GRAD_CACHE;
        const float amax = UNIT_BOUND ? 1.0f : global_amax[0];
        s_enc = UNIT_BOUND
            ? global_scale_denom
            : global_encode_scale(amax, global_scale_denom);
        if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
            row_sg_out[0] = UNIT_BOUND
                ? (1.0f / global_scale_denom)
                : (amax / global_scale_denom);
        }
        __syncthreads();
    }

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        uint8_t scales[8];

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            float vals[FP4_BLOCK];
            float block_amax = LOGSPACE_QUANT ? -INFINITY : 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float p = __bfloat162float(probs[r][b * FP4_BLOCK + j]);
                vals[j] = p;
                if constexpr (LOGSPACE_QUANT) {
                    block_amax = fmaxf(block_amax, p);
                } else {
                    block_amax = fmaxf(block_amax, GRAD_CACHE ? fabsf(p) : p);
                }
            }
            if constexpr (LOGSPACE_QUANT) {
                block_amax = isfinite(block_amax) ? __expf(block_amax) : 0.0f;
            }
            const float block_s_enc = CHUNK_SCALE && block_amax > 0.0f
                ? fminf(
                    s_enc,
                    NVFP4_GLOBAL_SCALE_DENOM / block_amax)
                : s_enc;
            const float scale_fold = CHUNK_SCALE
                ? (NVFP4_GLOBAL_SCALE_DENOM / block_s_enc)
                : 1.0f;
            uint8_t scale_byte = 0;
            float coeff;
            if constexpr (DIRECT_SCALE) {
                coeff = nvfp4_quant_coeff_direct_scale(
                    block_amax, block_s_enc, &scale_byte);
            } else {
                coeff = nvfp4_quant_coeff_opt<SCALE_SR>(
                    block_amax,
                    block_s_enc,
                    rng,
                    random_uint4,
                    rnd_idx,
                    &scale_byte,
                    scale_fold);
            }
            scales[b] = scale_byte;

            uint8_t packed_vals[8];
            if constexpr (LOGSPACE_QUANT) {
                pack_nvfp4_log_affine_block(
                    vals,
                    coeff,
                    logspace_affine_a,
                    logspace_affine_b,
                    packed_vals);
            } else {
                pack_nvfp4_block<ROW_DATA_SR>(
                    vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(row_fp4) + static_cast<int64_t>(gr) * (N / 2) + (col_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_nvfp4_scales(row_sc, tile_row, tile_col, N / 64, r, scales);
    } else if (tid < 2 * TILE) {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scales[8];

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            float vals[FP4_BLOCK];
            float block_amax = LOGSPACE_QUANT ? -INFINITY : 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float p = __bfloat162float(probs[b * FP4_BLOCK + j][c]);
                vals[j] = p;
                if constexpr (LOGSPACE_QUANT) {
                    block_amax = fmaxf(block_amax, p);
                } else {
                    block_amax = fmaxf(block_amax, GRAD_CACHE ? fabsf(p) : p);
                }
            }
            if constexpr (LOGSPACE_QUANT) {
                block_amax = isfinite(block_amax) ? __expf(block_amax) : 0.0f;
            }
            const float block_s_enc = CHUNK_SCALE && block_amax > 0.0f
                ? fminf(
                    s_enc,
                    NVFP4_GLOBAL_SCALE_DENOM / block_amax)
                : s_enc;
            const float scale_fold = CHUNK_SCALE
                ? (NVFP4_GLOBAL_SCALE_DENOM / block_s_enc)
                : 1.0f;
            uint8_t scale_byte = 0;
            float coeff;
            if constexpr (DIRECT_SCALE) {
                coeff = nvfp4_quant_coeff_direct_scale(
                    block_amax, block_s_enc, &scale_byte);
            } else {
                coeff = nvfp4_quant_coeff_opt<SCALE_SR>(
                    block_amax,
                    block_s_enc,
                    rng,
                    random_uint4,
                    rnd_idx,
                    &scale_byte,
                    scale_fold);
            }
            scales[b] = scale_byte;

            uint8_t packed_vals[8];
            if constexpr (LOGSPACE_QUANT) {
                pack_nvfp4_log_affine_block(
                    vals,
                    coeff,
                    logspace_affine_a,
                    logspace_affine_b,
                    packed_vals);
            } else {
                if constexpr (COL_ZERO_SR) {
                    pack_nvfp4_block_zero_sr(
                        vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
                } else {
                    pack_nvfp4_block<COL_DATA_SR>(
                        vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
                }
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) + static_cast<int64_t>(gc) * (M / 2) + (row_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_nvfp4_scales(col_sc, tile_col, tile_row, M / 64, c, scales);
    }
}

constexpr int WIDE_TILE_COLS = 256;
constexpr int WIDE_THREADS = TILE + WIDE_TILE_COLS;

// Cover two adjacent 128-column scale tiles in one CTA. This retains the exact
// row-wise and column-wise block-16 scale layout while halving CTA count and
// reusing each probability value for both orientations through shared memory.
__global__ __launch_bounds__(WIDE_THREADS)
void nvfp4_softmax_quant_row_col_wide_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    float* __restrict__ sg_out,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float global_scale_denom,
    float logit_temperature) {
    __shared__ __nv_bfloat16 probs[TILE][WIDE_TILE_COLS + 1];

    const int tile_col256 = blockIdx.x;
    const int tile_row128 = blockIdx.y;
    const int row_base = tile_row128 * TILE;
    const int col_base = tile_col256 * WIDE_TILE_COLS;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < TILE * WIDE_TILE_COLS; idx += blockDim.x) {
        const int r = idx / WIDE_TILE_COLS;
        const int c = idx - r * WIDE_TILE_COLS;
        const int gr = row_base + r;
        const int gc = col_base + c;
        float p = 0.0f;
        if (gr < M && gc < vocab_size && valid[gr]) {
            const float z = logit_temperature *
                __bfloat162float(logits[static_cast<int64_t>(gr) * N + gc]);
            p = __expf(z - lse[gr]);
        }
        probs[r][c] = __float2bfloat16(p);
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        sg_out[0] = 1.0f / global_scale_denom;
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint8_t scales[8];
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                float vals[FP4_BLOCK];
                float block_amax = 0.0f;
                const int local_col = half * TILE + b * FP4_BLOCK;
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; ++j) {
                    const float p = __bfloat162float(probs[r][local_col + j]);
                    vals[j] = p;
                    block_amax = fmaxf(block_amax, p);
                }
                uint8_t scale_byte = 0;
                const float coeff = nvfp4_quant_coeff(
                    block_amax, global_scale_denom, &scale_byte);
                scales[b] = scale_byte;
                uint8_t packed_vals[8];
                CceRNGState rng;
                uint4 random_uint4{};
                int rnd_idx = 4;
                pack_nvfp4_block<false>(
                    vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
                store_8_packed_fp4(
                    reinterpret_cast<uint8_t*>(row_fp4) +
                        static_cast<int64_t>(gr) * (N / 2) +
                        (col_base + local_col) / 2,
                    packed_vals);
            }
            store_nvfp4_scales(
                row_sc,
                tile_row128,
                tile_col256 * 2 + half,
                N / 64,
                r,
                scales);
        }
    } else {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scales[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            float vals[FP4_BLOCK];
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float p = __bfloat162float(probs[b * FP4_BLOCK + j][c]);
                vals[j] = p;
                block_amax = fmaxf(block_amax, p);
            }
            uint8_t scale_byte = 0;
            const float coeff = nvfp4_quant_coeff(
                block_amax, global_scale_denom, &scale_byte);
            scales[b] = scale_byte;
            uint8_t packed_vals[8];
            CceRNGState rng;
            uint4 random_uint4{};
            int rnd_idx = 4;
            pack_nvfp4_block<false>(
                vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(gc) * (M / 2) +
                    (row_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_nvfp4_scales(
            col_sc,
            tile_col256 * 2 + (c / TILE),
            tile_row128,
            M / 64,
            c & (TILE - 1),
            scales);
    }
}

template <int ROW_QUANT_THREADS>
__global__ __launch_bounds__(ROW_QUANT_THREADS)
void nvfp4_softmax_row_quant_lse_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    float* __restrict__ sg_out,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    int M,
    int N,
    int vocab_size,
    float global_scale_denom,
    float logit_temperature) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;
    if (row == 0 && tid == 0) {
        sg_out[0] = 1.0f / global_scale_denom;
    }

    const bool is_valid = valid[row];
    const float row_lse = lse[row];
    const int blocks = N / FP4_BLOCK;
    const int tile_row = row / TILE;
    const int row_in_tile = row & (TILE - 1);
    const int ntk64 = N / 64;

    for (int block = tid; block < blocks; block += ROW_QUANT_THREADS) {
        const int col0 = block * FP4_BLOCK;
        float vals[FP4_BLOCK];
        float block_amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < FP4_BLOCK; ++j) {
            const int col = col0 + j;
            float p = 0.0f;
            if (is_valid && col < vocab_size) {
                const float z = logit_temperature *
                    __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
                p = __bfloat162float(__float2bfloat16(__expf(z - row_lse)));
            }
            vals[j] = p;
            block_amax = fmaxf(block_amax, p);
        }
        uint8_t scale_byte = 0;
        const float coeff = nvfp4_quant_coeff(
            block_amax, global_scale_denom, &scale_byte);
        const int tile_col = col0 / TILE;
        const int scale_idx = (col0 / FP4_BLOCK) & 7;
        store_nvfp4_scale_one(
            row_sc,
            tile_row,
            tile_col,
            ntk64,
            row_in_tile,
            scale_idx,
            scale_byte);
        uint8_t packed_vals[8];
        CceRNGState rng;
        uint4 random_uint4{};
        int rnd_idx = 4;
        pack_nvfp4_block<false>(
            vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
        store_8_packed_fp4(
            reinterpret_cast<uint8_t*>(row_fp4) +
                static_cast<int64_t>(row) * (N / 2) + col0 / 2,
            packed_vals);
    }
}

template <
    bool APPLY_ROW_NORMALIZATION,
    bool PACKED_SHARED_STORES = false,
    bool COL_DATA_SR = false,
    bool COL_ZERO_SR = false>
__global__ __launch_bounds__(THREADS)
void nvfp4_col_requant_from_row_kernel(
    const __nv_fp4x2_e2m1* __restrict__ row_fp4,
    const uint8_t* __restrict__ row_sc,
    const float* __restrict__ row_normalization,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    float global_scale_denom,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    const uint64_t* __restrict__ rng_state = nullptr) {
    __shared__ __nv_bfloat16 probs[
        TILE][PACKED_SHARED_STORES ? TILE + 2 : TILE + 1];
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;
    const int ntk64_row = N / 64;

    CceRNGState rng;
    uint4 random_uint4{};
    int rnd_idx = 4;
    if constexpr (COL_DATA_SR || COL_ZERO_SR) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
        const uint64_t tile_sequence =
            (static_cast<uint64_t>(tile_row) * gridDim.x + tile_col) * 256ull;
        rng.init(rng_seed, rng_subsequence_base + tile_sequence + tid, 0);
    }

    for (int block = tid; block < TILE * (TILE / FP4_BLOCK);
         block += blockDim.x) {
        const int r = block / (TILE / FP4_BLOCK);
        const int b = block & ((TILE / FP4_BLOCK) - 1);
        const int gr = row_base + r;
        const uint8_t scale_raw = load_nvfp4_scale_one(
            row_sc, tile_row, tile_col, ntk64_row, r, b);
        const float represented_scale = e4m3_from_byte(scale_raw) *
            (APPLY_ROW_NORMALIZATION ? row_normalization[gr] : 1.0f);
        const int64_t packed_base =
            static_cast<int64_t>(gr) * (N / 2) + (col_base / 2) + b * 8;
        // Block-16 storage is eight aligned FP4x2 bytes.
        const uint64_t packed_values = *reinterpret_cast<const uint64_t*>(
            row_fp4 + packed_base);
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            __nv_fp4x2_e2m1 packed_pair;
            packed_pair.__x = static_cast<__nv_fp4x2_storage_t>(
                packed_values >> (pair * 8));
            const float2 q = static_cast<float2>(packed_pair);
            if constexpr (PACKED_SHARED_STORES) {
                *reinterpret_cast<__nv_bfloat162*>(
                    &probs[r][b * FP4_BLOCK + pair * 2]) =
                    __floats2bfloat162_rn(
                        q.x * represented_scale,
                        q.y * represented_scale);
            } else {
                probs[r][b * FP4_BLOCK + pair * 2] =
                    __float2bfloat16(q.x * represented_scale);
                probs[r][b * FP4_BLOCK + pair * 2 + 1] =
                    __float2bfloat16(q.y * represented_scale);
            }
        }
    }
    __syncthreads();

    if (tid < TILE) {
        const int c = tid;
        const int gc = col_base + c;
        uint8_t scales[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            float vals[FP4_BLOCK];
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float p = __bfloat162float(probs[b * FP4_BLOCK + j][c]);
                vals[j] = p;
                block_amax = fmaxf(block_amax, p);
            }
            uint8_t scale_byte = 0;
            const float coeff = nvfp4_quant_coeff(
                block_amax, 1.0f, &scale_byte);
            scales[b] = scale_byte;
            uint8_t packed_vals[8];
            if constexpr (COL_ZERO_SR) {
                pack_nvfp4_block_zero_sr(
                    vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            } else {
                pack_nvfp4_block<COL_DATA_SR>(
                    vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(gc) * (M / 2) +
                    (row_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_nvfp4_scales(
            col_sc, tile_col, tile_row, M / 64, c, scales);
    }
}

template <bool COL_DATA_SR = false, bool COL_ZERO_SR = false>
__global__ __launch_bounds__(THREADS)
void nvfp4_col_requant_from_mxfp8_row_kernel(
    const uint8_t* __restrict__ row_fp8,
    const uint8_t* __restrict__ row_sc,
    const float* __restrict__ row_normalization,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    float* __restrict__ col_sg,
    int M,
    int N,
    float global_scale_denom,
    uint64_t rng_seed = 0,
    uint64_t rng_subsequence_base = 0,
    const uint64_t* __restrict__ rng_state = nullptr) {
    __shared__ __nv_bfloat16 probs[TILE][TILE + 1];
    constexpr int MXFP8_BLOCK = 32;
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;
    const int tiles_per_row = N / TILE;

    CceRNGState rng;
    uint4 random_uint4{};
    int rnd_idx = 4;
    if constexpr (COL_DATA_SR || COL_ZERO_SR) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
        const uint64_t tile_sequence =
            (static_cast<uint64_t>(tile_row) * gridDim.x + tile_col) * 256ull;
        rng.init(rng_seed, rng_subsequence_base + tile_sequence + tid, 0);
    }

    for (int block = tid; block < TILE * (TILE / MXFP8_BLOCK);
         block += blockDim.x) {
        const int r = block / (TILE / MXFP8_BLOCK);
        const int b = block & ((TILE / MXFP8_BLOCK) - 1);
        const int gr = row_base + r;
        const int scale_offset =
            (tile_row * tiles_per_row + tile_col) * 512 +
            (r & 31) * 16 + (r >> 5) * 4 + b;
        const float represented_scale = e8m0_from_byte(row_sc[scale_offset]) *
            row_normalization[gr];
        const int64_t input_offset =
            static_cast<int64_t>(gr) * N + col_base + b * MXFP8_BLOCK;
        const auto* packed_input = reinterpret_cast<const uint4*>(
            row_fp8 + input_offset);
        const uint4 first = packed_input[0];
        const uint4 second = packed_input[1];
        const auto* input_bytes = reinterpret_cast<const uint8_t*>(&first);
        const auto* second_bytes = reinterpret_cast<const uint8_t*>(&second);
        #pragma unroll
        for (int j = 0; j < MXFP8_BLOCK; ++j) {
            const uint8_t raw = j < 16 ? input_bytes[j] : second_bytes[j - 16];
            probs[r][b * MXFP8_BLOCK + j] =
                __float2bfloat16(e4m3_from_byte(raw) * represented_scale);
        }
    }
    if (tile_row == 0 && tile_col == 0 && tid == 0) {
        col_sg[0] = 1.0f / global_scale_denom;
    }
    __syncthreads();

    if (tid < TILE) {
        const int c = tid;
        const int gc = col_base + c;
        uint8_t scales[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            float vals[FP4_BLOCK];
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float value = __bfloat162float(
                    probs[b * FP4_BLOCK + j][c]);
                vals[j] = value;
                block_amax = fmaxf(block_amax, fabsf(value));
            }
            uint8_t scale_byte = 0;
            const float coeff = nvfp4_quant_coeff(
                block_amax, global_scale_denom, &scale_byte);
            scales[b] = scale_byte;
            uint8_t packed_vals[8];
            if constexpr (COL_ZERO_SR) {
                pack_nvfp4_block_zero_sr(
                    vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            } else {
                pack_nvfp4_block<COL_DATA_SR>(
                    vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(gc) * (M / 2) +
                    (row_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_nvfp4_scales(
            col_sc, tile_col, tile_row, M / 64, c, scales);
    }
}

// Produce row and transposed block-16 operands directly from warp registers.
// Two lanes own each 16-value row; lanes with the same parity own one set of
// eight columns. XOR reductions therefore recover both orientation-specific
// maxima without a shared transpose tile or CTA barrier.
__global__ __launch_bounds__(THREADS)
void nvfp4_softmax_quant_row_col_register_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    float* __restrict__ sg_out,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float global_scale_denom,
    float logit_temperature) {
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = lane >> 1;
    const int half = lane & 1;
    constexpr unsigned FULL_MASK = 0xffffffffu;
    __shared__ uint8_t packed_col_bytes[TILE][TILE / 2 + 1];
    __shared__ uint8_t row_scales[TILE][TILE / FP4_BLOCK];
    __shared__ uint8_t col_scales[TILE][TILE / FP4_BLOCK];

    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        sg_out[0] = 1.0f / global_scale_denom;
    }

    #pragma unroll
    for (int subtile = warp; subtile < 64; subtile += THREADS / 32) {
        const int subtile_r = subtile >> 3;
        const int subtile_c = subtile & 7;
        const int local_row = subtile_r * FP4_BLOCK + row;
        const int local_col = subtile_c * FP4_BLOCK + half * 8;
        const int global_row = row_base + local_row;
        const int global_col = col_base + local_col;
        const bool row_valid = global_row < M && valid[global_row];
        const float row_lse = row_valid ? lse[global_row] : 0.0f;

        float vals[8];
        float row_amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float p = 0.0f;
            const int gc = global_col + j;
            if (row_valid && gc < vocab_size) {
                const float z = logit_temperature * __bfloat162float(
                    logits[static_cast<int64_t>(global_row) * N + gc]);
                // Match the existing tiled producer's BF16 shared staging.
                p = __bfloat162float(__float2bfloat16(__expf(z - row_lse)));
            }
            vals[j] = p;
            row_amax = fmaxf(row_amax, p);
        }
        row_amax = fmaxf(
            row_amax, __shfl_xor_sync(FULL_MASK, row_amax, 1));

        float row_coeff = 0.0f;
        uint8_t row_scale = 0;
        if (half == 0) {
            row_coeff = nvfp4_quant_coeff(
                row_amax, global_scale_denom, &row_scale);
        }
        row_coeff = __shfl_sync(FULL_MASK, row_coeff, lane & ~1);
        row_scale = static_cast<uint8_t>(__shfl_sync(
            FULL_MASK, static_cast<int>(row_scale), lane & ~1));

        uint8_t row_bytes[4];
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float2 scaled = {
                vals[j] * row_coeff,
                vals[j + 1] * row_coeff,
            };
            row_bytes[j >> 1] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(
                    scaled, __NV_E2M1, cudaRoundNearest));
        }
        const uint32_t packed_row =
            static_cast<uint32_t>(row_bytes[0]) |
            (static_cast<uint32_t>(row_bytes[1]) << 8) |
            (static_cast<uint32_t>(row_bytes[2]) << 16) |
            (static_cast<uint32_t>(row_bytes[3]) << 24);
        *reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(row_fp4) +
            static_cast<int64_t>(global_row) * (N / 2) + global_col / 2) =
                packed_row;
        if (half == 0) {
            row_scales[local_row][subtile_c] = row_scale;
        }

        float col_coeff[8];
        uint8_t col_scale[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float col_amax = vals[j];
            col_amax = fmaxf(
                col_amax, __shfl_xor_sync(FULL_MASK, col_amax, 2));
            col_amax = fmaxf(
                col_amax, __shfl_xor_sync(FULL_MASK, col_amax, 4));
            col_amax = fmaxf(
                col_amax, __shfl_xor_sync(FULL_MASK, col_amax, 8));
            col_amax = fmaxf(
                col_amax, __shfl_xor_sync(FULL_MASK, col_amax, 16));
            float coeff = 0.0f;
            uint8_t scale = 0;
            if (row == 0) {
                coeff = nvfp4_quant_coeff(
                    col_amax, global_scale_denom, &scale);
            }
            col_coeff[j] = __shfl_sync(FULL_MASK, coeff, half);
            col_scale[j] = static_cast<uint8_t>(__shfl_sync(
                FULL_MASK, static_cast<int>(scale), half));
        }

        float next_row_vals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            next_row_vals[j] = __shfl_down_sync(FULL_MASK, vals[j], 2);
        }
        if ((row & 1) == 0) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 scaled = {
                    vals[j] * col_coeff[j],
                    next_row_vals[j] * col_coeff[j],
                };
                const uint8_t packed_col_byte = static_cast<uint8_t>(
                    __nv_cvt_float2_to_fp4x2(
                        scaled, __NV_E2M1, cudaRoundNearest));
                packed_col_bytes[local_col + j][local_row / 2] =
                    packed_col_byte;
            }
        }
        if (row == 0) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                col_scales[local_col + j][subtile_r] = col_scale[j];
            }
        }
    }
    __syncthreads();

    if (tid < TILE) {
        const int c = tid;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            uint8_t bytes[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                bytes[j] = packed_col_bytes[c][b * 8 + j];
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(col_base + c) * (M / 2) +
                    (row_base + b * FP4_BLOCK) / 2,
                bytes);
        }
        uint8_t scale_bytes[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            scale_bytes[b] = row_scales[c][b];
        }
        store_nvfp4_scales(
            row_sc, tile_row, tile_col, N / 64, c, scale_bytes);
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            scale_bytes[b] = col_scales[c][b];
        }
        store_nvfp4_scales(
            col_sc, tile_col, tile_row, M / 64, c, scale_bytes);
    }
}

// Use one scale for each 16x16 probability tile. The shared scale makes the
// quantized value identical in the row and transposed operands, so conversion
// is performed once and only the packed nibbles are transposed for the second
// store. This is a diagnostic tail format; selected probabilities are repaired
// separately by the CCE top-k path.
__global__ __launch_bounds__(THREADS)
void nvfp4_softmax_quant_row_col_2d_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    float* __restrict__ sg_out,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float global_scale_denom,
    float logit_temperature) {
    __shared__ __nv_bfloat16 probs[TILE][TILE + 1];
    __shared__ uint8_t packed[TILE][TILE / 2 + 1];
    __shared__ uint8_t scales[TILE / FP4_BLOCK][TILE / FP4_BLOCK + 1];

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
        const int r = idx / TILE;
        const int c = idx - r * TILE;
        const int gr = row_base + r;
        const int gc = col_base + c;
        float p = 0.0f;
        if (gr < M && gc < vocab_size && valid[gr]) {
            const float z = logit_temperature *
                __bfloat162float(logits[static_cast<int64_t>(gr) * N + gc]);
            p = __expf(z - lse[gr]);
        }
        probs[r][c] = __float2bfloat16(p);
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        sg_out[0] = 1.0f / global_scale_denom;
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int subtile = warp; subtile < 64; subtile += THREADS / 32) {
        const int subtile_r = subtile >> 3;
        const int subtile_c = subtile & 7;
        const int half_row = lane >> 1;
        const int half_col = (lane & 1) * 8;
        float vals[8];
        float local_amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p = __bfloat162float(
                probs[subtile_r * FP4_BLOCK + half_row]
                     [subtile_c * FP4_BLOCK + half_col + j]);
            vals[j] = p;
            local_amax = fmaxf(local_amax, p);
        }
        const float block_amax = warp_reduce_max(local_amax);
        float coeff = 0.0f;
        uint8_t scale_byte = 0;
        if (lane == 0) {
            coeff = nvfp4_quant_coeff(
                block_amax, global_scale_denom, &scale_byte);
            scales[subtile_r][subtile_c] = scale_byte;
        }
        coeff = __shfl_sync(0xffffffff, coeff, 0);

        uint8_t bytes[4];
        #pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float2 scaled = {
                vals[j] * coeff,
                vals[j + 1] * coeff,
            };
            bytes[j >> 1] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(
                    scaled, __NV_E2M1, cudaRoundNearest));
        }
        const int packed_row = subtile_r * FP4_BLOCK + half_row;
        const int packed_col = subtile_c * (FP4_BLOCK / 2) +
            (lane & 1) * (FP4_BLOCK / 4);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            packed[packed_row][packed_col + j] = bytes[j];
        }
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        uint8_t scale_bytes[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            uint8_t bytes[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                bytes[j] = packed[r][b * 8 + j];
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(row_fp4) +
                    static_cast<int64_t>(row_base + r) * (N / 2) +
                    (col_base + b * FP4_BLOCK) / 2,
                bytes);
            scale_bytes[b] = scales[r / FP4_BLOCK][b];
        }
        store_nvfp4_scales(
            row_sc, tile_row, tile_col, N / 64, r, scale_bytes);
    } else {
        const int c = tid - TILE;
        uint8_t scale_bytes[8];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            uint8_t bytes[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r0 = b * FP4_BLOCK + 2 * j;
                const int r1 = r0 + 1;
                const uint8_t raw0 = packed[r0][c >> 1];
                const uint8_t raw1 = packed[r1][c >> 1];
                const int shift = (c & 1) * 4;
                bytes[j] = static_cast<uint8_t>(
                    ((raw0 >> shift) & 0x0f) |
                    (((raw1 >> shift) & 0x0f) << 4));
            }
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(col_base + c) * (M / 2) +
                    (row_base + b * FP4_BLOCK) / 2,
                bytes);
            scale_bytes[b] = scales[b][c / FP4_BLOCK];
        }
        store_nvfp4_scales(
            col_sc, tile_col, tile_row, M / 64, c, scale_bytes);
    }
}

__device__ __forceinline__ void store_mxfp8_scales(
    uint8_t* __restrict__ scales,
    int tile_row,
    int tile_col,
    int tiles_per_row,
    int row_in_tile,
    const uint8_t vals[4]) {
    const int scale_offset =
        (tile_row * tiles_per_row + tile_col) * 512 +
        (row_in_tile & 31) * 16 +
        (row_in_tile >> 5) * 4;
    const uint32_t packed =
        static_cast<uint32_t>(vals[0]) |
        (static_cast<uint32_t>(vals[1]) << 8) |
        (static_cast<uint32_t>(vals[2]) << 16) |
        (static_cast<uint32_t>(vals[3]) << 24);
    *reinterpret_cast<uint32_t*>(scales + scale_offset) = packed;
}

__global__ __launch_bounds__(THREADS)
void mxfp8_col_requant_from_mxfp8_row_kernel(
    const uint8_t* __restrict__ row_fp8,
    const uint8_t* __restrict__ row_sc,
    const float* __restrict__ row_normalization,
    uint8_t* __restrict__ col_fp8,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    float quant_max) {
    // Decode the existing numerator row operand once, fold in its delayed row
    // normalization, and reuse a padded BF16 tile for the transpose requant.
    __shared__ __nv_bfloat16 values[TILE][TILE + 1];
    constexpr int MXFP8_BLOCK = 32;

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;
    const int tiles_per_row = N / TILE;

    for (int block = tid; block < TILE * (TILE / MXFP8_BLOCK);
         block += blockDim.x) {
        const int r = block / (TILE / MXFP8_BLOCK);
        const int b = block & ((TILE / MXFP8_BLOCK) - 1);
        const int gr = row_base + r;
        const int scale_offset =
            (tile_row * tiles_per_row + tile_col) * 512 +
            (r & 31) * 16 + (r >> 5) * 4 + b;
        const float represented_scale = e8m0_from_byte(row_sc[scale_offset]) *
            row_normalization[gr];
        const int64_t input_offset =
            static_cast<int64_t>(gr) * N + col_base + b * MXFP8_BLOCK;
        const auto* packed_input = reinterpret_cast<const uint4*>(
            row_fp8 + input_offset);
        const uint4 first = packed_input[0];
        const uint4 second = packed_input[1];
        const auto* first_bytes = reinterpret_cast<const uint8_t*>(&first);
        const auto* second_bytes = reinterpret_cast<const uint8_t*>(&second);
        #pragma unroll
        for (int j = 0; j < MXFP8_BLOCK; ++j) {
            const uint8_t raw =
                j < 16 ? first_bytes[j] : second_bytes[j - 16];
            values[r][b * MXFP8_BLOCK + j] = __float2bfloat16(
                e4m3_from_byte(raw) * represented_scale);
        }
    }
    __syncthreads();

    if (tid < TILE) {
        const int c = tid;
        const int gc = col_base + c;
        uint8_t scale_bytes[4];

        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < MXFP8_BLOCK; ++j) {
                const float value = __bfloat162float(
                    values[block * MXFP8_BLOCK + j][c]);
                block_amax = fmaxf(block_amax, fabsf(value));
            }

            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax / quant_max, 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_bytes[block] = scale.__x;
            const float scale_inv = 1.0f / static_cast<float>(scale);

            uint4 packed_output[2]{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
            #pragma unroll
            for (int j = 0; j < MXFP8_BLOCK; ++j) {
                const float value = __bfloat162float(
                    values[block * MXFP8_BLOCK + j][c]);
                const __nv_fp8_e4m3 quantized(value * scale_inv);
                output_bytes[j] = quantized.__x;
            }
            auto* output_ptr =
                col_fp8 + static_cast<int64_t>(gc) * M +
                row_base + block * MXFP8_BLOCK;
            reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
            reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
        }
        store_mxfp8_scales(
            col_sc, tile_col, tile_row, M / TILE, c, scale_bytes);
    }
}

__device__ __forceinline__ uint8_t mxfp4_encode_ceil_amax(float value) {
    if (value <= 1.0e-38f) {
        return 0;
    }
    const uint32_t bits = __float_as_uint(value);
    uint8_t exponent = static_cast<uint8_t>((bits >> 23) & 0xff);
    if ((bits & 0x7fffff) != 0 && exponent < 0xfe) {
        ++exponent;
    }
    return exponent;
}

__device__ __forceinline__ float mxfp4_reciprocal_e8m0(uint8_t exponent) {
    if (exponent == 0xff) {
        return __int_as_float(0x7fffffff);
    }
    if (exponent == 0xfe) {
        return __int_as_float(0x00400000);
    }
    return __int_as_float((254 - static_cast<int>(exponent)) << 23);
}

// Emit the two classifier-weight contracts needed by the mixed backward:
// MXFP4 rows for logits and MXFP8(input.T) rows for dE.
__global__ __launch_bounds__(THREADS)
void mxfp4_row_mxfp8_col_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    uint8_t* __restrict__ col_fp8,
    uint8_t* __restrict__ col_sc,
    int M,
    int N) {
    __shared__ __nv_bfloat16 values[TILE][TILE + 1];

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
        const int r = idx / TILE;
        const int c = idx - r * TILE;
        values[r][c] = input[
            static_cast<int64_t>(row_base + r) * N + col_base + c];
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        uint8_t scale_bytes[4];
        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                block_amax = fmaxf(
                    block_amax,
                    fabsf(__bfloat162float(values[r][block * 32 + j])));
            }
            const uint8_t scale = mxfp4_encode_ceil_amax(block_amax);
            scale_bytes[block] = scale;
            const float coefficient = 6.0f * mxfp4_reciprocal_e8m0(scale);

            uint4 packed{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(&packed);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float2 pair = {
                    __bfloat162float(values[r][block * 32 + 2 * j]) * coefficient,
                    __bfloat162float(values[r][block * 32 + 2 * j + 1]) * coefficient,
                };
                output_bytes[j] = static_cast<uint8_t>(
                    __nv_cvt_float2_to_fp4x2(
                        pair, __NV_E2M1, cudaRoundNearest));
            }
            auto* output_ptr = reinterpret_cast<uint8_t*>(row_fp4) +
                static_cast<int64_t>(gr) * (N / 2) +
                (col_base + block * 32) / 2;
            *reinterpret_cast<uint4*>(output_ptr) = packed;
        }
        store_mxfp8_scales(
            row_sc, tile_row, tile_col, N / TILE, r, scale_bytes);
    } else {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scale_bytes[4];
        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                block_amax = fmaxf(
                    block_amax,
                    fabsf(__bfloat162float(values[block * 32 + j][c])));
            }
            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax / 448.0f, 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_bytes[block] = scale.__x;
            const float scale_inv = 1.0f / static_cast<float>(scale);

            uint4 packed_output[2]{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const __nv_fp8_e4m3 quantized(
                    __bfloat162float(values[block * 32 + j][c]) * scale_inv);
                output_bytes[j] = quantized.__x;
            }
            auto* output_ptr = col_fp8 +
                static_cast<int64_t>(gc) * M + row_base + block * 32;
            reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
            reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
        }
        store_mxfp8_scales(
            col_sc, tile_col, tile_row, M / TILE, c, scale_bytes);
    }
}

// Quantize the selected-masked softmax tail once into an MXFP8 row operand
// for dE and a native NVFP4 column operand for dW.
__global__ __launch_bounds__(THREADS)
void mxfp8_row_nvfp4_col_softmax_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    uint8_t* __restrict__ row_fp8,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    float* __restrict__ col_sg,
    int M,
    int N,
    int vocab_size,
    float mxfp8_quant_max,
    float nvfp4_global_scale_denom,
    float logit_temperature) {
    __shared__ __nv_bfloat16 probs[TILE][TILE + 1];

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
        const int r = idx / TILE;
        const int c = idx - r * TILE;
        const int gr = row_base + r;
        const int gc = col_base + c;
        float p = 0.0f;
        if (gr < M && gc < vocab_size && valid[gr]) {
            const float z = logit_temperature * __bfloat162float(
                logits[static_cast<int64_t>(gr) * N + gc]);
            p = __expf(z - lse[gr]);
        }
        probs[r][c] = __float2bfloat16(p);
    }
    if (tile_row == 0 && tile_col == 0 && tid == 0) {
        col_sg[0] = 1.0f / nvfp4_global_scale_denom;
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        uint8_t scale_bytes[4];
        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                block_amax = fmaxf(
                    block_amax,
                    __bfloat162float(probs[r][block * 32 + j]));
            }
            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax / mxfp8_quant_max, 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_bytes[block] = scale.__x;
            const float scale_inv = 1.0f / static_cast<float>(scale);

            uint4 packed_output[2]{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const __nv_fp8_e4m3 quantized(
                    __bfloat162float(probs[r][block * 32 + j]) * scale_inv);
                output_bytes[j] = quantized.__x;
            }
            auto* output_ptr = row_fp8 +
                static_cast<int64_t>(gr) * N + col_base + block * 32;
            reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
            reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
        }
        store_mxfp8_scales(
            row_sc, tile_row, tile_col, N / TILE, r, scale_bytes);
    } else {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scale_bytes[8];
        #pragma unroll
        for (int block = 0; block < 8; ++block) {
            float vals[FP4_BLOCK];
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float p = __bfloat162float(
                    probs[block * FP4_BLOCK + j][c]);
                vals[j] = p;
                block_amax = fmaxf(block_amax, p);
            }
            uint8_t scale_byte = 0;
            const float coeff = nvfp4_quant_coeff(
                block_amax, nvfp4_global_scale_denom, &scale_byte);
            scale_bytes[block] = scale_byte;
            uint8_t packed_vals[8];
            CceRNGState rng;
            uint4 random_uint4{};
            int rnd_idx = 4;
            pack_nvfp4_block<false>(
                vals, coeff, rng, random_uint4, rnd_idx, packed_vals);
            store_8_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(gc) * (M / 2) +
                    (row_base + block * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_nvfp4_scales(
            col_sc, tile_col, tile_row, M / 64, c, scale_bytes);
    }
}

template <
    bool SOFTMAX_INPUT,
    bool RMSNORM_INPUT = false,
    bool WRITE_COLUMN = true>
__global__ __launch_bounds__(THREADS)
void mxfp8_quant_row_col_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    __nv_bfloat16* __restrict__ normed,
    uint8_t* __restrict__ row_fp8,
    uint8_t* __restrict__ row_sc,
    uint8_t* __restrict__ col_fp8,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float quant_max) {
    // One exponentiation feeds both orientations. Padding avoids shared-memory
    // bank conflicts in the transpose path.
    __shared__ __nv_bfloat16 values[TILE][TILE + 1];

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
        const int r = idx / TILE;
        const int c = idx - r * TILE;
        const int gr = row_base + r;
        const int gc = col_base + c;

        float value = 0.0f;
        if constexpr (SOFTMAX_INPUT) {
            if (gr < M && gc < vocab_size && valid[gr]) {
                const float z = __bfloat162float(
                    input[static_cast<int64_t>(gr) * N + gc]);
                value = __expf(z - lse[gr]);
            }
        } else if constexpr (RMSNORM_INPUT) {
            if (gr < M && gc < N) {
                value =
                    __bfloat162float(input[static_cast<int64_t>(gr) * N + gc]) *
                    inv_rms[gr] * __bfloat162float(gamma[gc]);
                normed[static_cast<int64_t>(gr) * N + gc] =
                    __float2bfloat16(value);
            }
        } else if (gr < M && gc < N) {
            value = __bfloat162float(
                input[static_cast<int64_t>(gr) * N + gc]);
        }
        values[r][c] = __float2bfloat16(value);
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        uint8_t scale_bytes[4];

        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float value = __bfloat162float(values[r][block * 32 + j]);
                block_amax = fmaxf(block_amax, fabsf(value));
            }

            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax / quant_max, 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_bytes[block] = scale.__x;
            const float scale_inv = 1.0f / static_cast<float>(scale);

            uint4 packed_output[2]{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float value = __bfloat162float(values[r][block * 32 + j]);
                const __nv_fp8_e4m3 quantized(value * scale_inv);
                output_bytes[j] = quantized.__x;
            }
            auto* output_ptr =
                row_fp8 + static_cast<int64_t>(gr) * N + col_base + block * 32;
            reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
            reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
        }
        store_mxfp8_scales(
            row_sc, tile_row, tile_col, N / TILE, r, scale_bytes);
    } else if constexpr (WRITE_COLUMN) {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scale_bytes[4];

        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float value = __bfloat162float(values[block * 32 + j][c]);
                block_amax = fmaxf(block_amax, fabsf(value));
            }

            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax / quant_max, 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_bytes[block] = scale.__x;
            const float scale_inv = 1.0f / static_cast<float>(scale);

            uint4 packed_output[2]{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float value = __bfloat162float(values[block * 32 + j][c]);
                const __nv_fp8_e4m3 quantized(value * scale_inv);
                output_bytes[j] = quantized.__x;
            }
            auto* output_ptr =
                col_fp8 + static_cast<int64_t>(gc) * M + row_base + block * 32;
            reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
            reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
        }
        store_mxfp8_scales(
            col_sc, tile_col, tile_row, M / TILE, c, scale_bytes);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp8_quant_row(torch::Tensor input, double quant_max) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
        "input must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    const int64_t M64 = input.size(0);
    const int64_t N64 = input.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "MXFP8 row quantization requires dimensions divisible by 128");

    auto device = input.device();
    auto fp8_options = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto scale_options = torch::dtype(torch::kUInt8).device(device);
    auto row_fp8 = torch::empty({M64, N64}, fp8_options);
    auto row_sc = torch::empty(
        {M64 / TILE, N64 / TILE, 32, 16}, scale_options);

    // Keep the full producer's launch geometry and row-side instructions
    // exactly unchanged.  The compile-time gate only suppresses the
    // transpose writes and, critically, their payload/scale allocations.
    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_quant_row_col_kernel<false, false, false>
        <<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
            row_sc.data_ptr<uint8_t>(),
            nullptr,
            nullptr,
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(N64),
            static_cast<float>(quant_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_fp8, row_sc);
}

__global__ __launch_bounds__(THREADS)
void mxfp8_quant_col_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ col_fp8,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    float quant_max) {
    // This is the column half of mxfp8_quant_row_col_kernel.  Keep the BF16
    // staging and quantization arithmetic identical so this standalone pass
    // emits the same native transposed operand without replacing the deployed
    // first-stage row producer.
    __shared__ __nv_bfloat16 values[TILE][TILE + 1];

    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < TILE * TILE; idx += blockDim.x) {
        const int r = idx / TILE;
        const int c = idx - r * TILE;
        const int gr = row_base + r;
        const int gc = col_base + c;
        values[r][c] = __float2bfloat16(__bfloat162float(
            input[static_cast<int64_t>(gr) * N + gc]));
    }
    __syncthreads();

    if (tid < TILE) {
        const int c = tid;
        const int gc = col_base + c;
        uint8_t scale_bytes[4];

        #pragma unroll
        for (int block = 0; block < 4; ++block) {
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float value = __bfloat162float(
                    values[block * 32 + j][c]);
                block_amax = fmaxf(block_amax, fabsf(value));
            }

            __nv_fp8_e8m0 scale;
            scale.__x = __nv_cvt_float_to_e8m0(
                fmaxf(block_amax / quant_max, 1.0e-12f),
                __NV_SATFINITE,
                cudaRoundPosInf);
            scale_bytes[block] = scale.__x;
            const float scale_inv = 1.0f / static_cast<float>(scale);

            uint4 packed_output[2]{};
            auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float value = __bfloat162float(
                    values[block * 32 + j][c]);
                const __nv_fp8_e4m3 quantized(value * scale_inv);
                output_bytes[j] = quantized.__x;
            }
            auto* output_ptr =
                col_fp8 + static_cast<int64_t>(gc) * M +
                row_base + block * 32;
            reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
            reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
        }
        store_mxfp8_scales(
            col_sc, tile_col, tile_row, M / TILE, c, scale_bytes);
    }
}

__global__ __launch_bounds__(THREADS)
void mxfp8_rmsnorm_inv_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ inv_rms,
    int M,
    int N,
    float epsilon) {
    const int row = blockIdx.x;
    if (row >= M) return;
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        const float value = __bfloat162float(
            input[static_cast<int64_t>(row) * N + col]);
        sum_sq = fmaf(value, value, sum_sq);
    }
    sum_sq = block_reduce_sum(sum_sq);
    if (threadIdx.x == 0) {
        inv_rms[row] = rsqrtf(sum_sq / static_cast<float>(N) + epsilon);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp8_quant_row_col(torch::Tensor input, double quant_max) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
        "input must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    const int64_t M64 = input.size(0);
    const int64_t N64 = input.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "MXFP8 row/col quantization requires dimensions divisible by 128");

    auto device = input.device();
    auto fp8_options = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto scale_options = torch::dtype(torch::kUInt8).device(device);
    auto row_fp8 = torch::empty({M64, N64}, fp8_options);
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, scale_options);
    auto col_fp8 = torch::empty({N64, M64}, fp8_options);
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, scale_options);

    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_quant_row_col_kernel<false><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(N64),
        static_cast<float>(quant_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_fp8, row_sc, col_fp8, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp8_quant_col(torch::Tensor input, double quant_max) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
        "input must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    const int64_t M64 = input.size(0);
    const int64_t N64 = input.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "MXFP8 column quantization requires dimensions divisible by 128");
    TORCH_CHECK(
        M64 <= std::numeric_limits<int>::max() &&
            N64 <= std::numeric_limits<int>::max(),
        "MXFP8 column quantization dimensions exceed int32 range");

    auto device = input.device();
    auto col_fp8 = torch::empty(
        {N64, M64},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_quant_col_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<uint8_t*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<float>(quant_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(col_fp8, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quant_row_mxfp8_col(torch::Tensor input) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
        "input must be contiguous CUDA BF16 [M, N]");

    const int64_t M64 = input.size(0);
    const int64_t N64 = input.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "mixed MXFP4/MXFP8 quantization requires dimensions divisible by 128");

    auto device = input.device();
    auto row_fp4 = torch::empty(
        {M64, N64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M64 / TILE, N64 / TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp8 = torch::empty(
        {N64, M64},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp4_row_mxfp8_col_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        static_cast<int>(M64),
        static_cast<int>(N64));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_fp4, row_sc, col_fp8, col_sc);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
mxfp8_rmsnorm_quant_row_col(
    torch::Tensor input,
    torch::Tensor gamma,
    double epsilon,
    double quant_max) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
            input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
        "input must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() &&
            gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
        "gamma must be contiguous CUDA BF16 [N]");
    TORCH_CHECK(gamma.size(0) == input.size(1), "gamma length mismatch");
    TORCH_CHECK(
        std::isfinite(epsilon) && epsilon >= 0.0,
        "epsilon must be finite and non-negative");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    const int64_t M64 = input.size(0);
    const int64_t N64 = input.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "MXFP8 RMSNorm row/col quantization requires dimensions divisible by 128");

    auto device = input.device();
    auto fp8_options = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto scale_options = torch::dtype(torch::kUInt8).device(device);
    auto float_options = torch::dtype(torch::kFloat32).device(device);
    auto normed = torch::empty_like(input);
    auto row_fp8 = torch::empty({M64, N64}, fp8_options);
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, scale_options);
    auto col_fp8 = torch::empty({N64, M64}, fp8_options);
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, scale_options);
    auto inv_rms = torch::empty({M64}, float_options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_rmsnorm_inv_kernel<<<M64, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<float>(epsilon));
    const dim3 grid(N64 / TILE, M64 / TILE);
    mxfp8_quant_row_col_kernel<false, true><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(normed.data_ptr()),
        reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(N64),
        static_cast<float>(quant_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(normed, row_fp8, row_sc, col_fp8, col_sc, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp8_softmax_quant_row_col(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    double quant_max) {
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2,
        "logits must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        lse.is_cuda() && lse.is_contiguous() &&
            lse.scalar_type() == torch::kFloat,
        "lse must be contiguous CUDA float");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "MXFP8 G-tail quantization requires dimensions divisible by 128");
    TORCH_CHECK(
        lse.numel() == M64 && valid.numel() == M64,
        "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto fp8_options = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto scale_options = torch::dtype(torch::kUInt8).device(device);
    auto row_fp8 = torch::empty({M64, N64}, fp8_options);
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, scale_options);
    auto col_fp8 = torch::empty({N64, M64}, fp8_options);
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, scale_options);

    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_quant_row_col_kernel<true><<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        lse.data_ptr<float>(),
        valid.data_ptr<bool>(),
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(vocab_size),
        static_cast<float>(quant_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_fp8, row_sc, col_fp8, col_sc);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
mxfp8_row_nvfp4_col_softmax_quant(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    double mxfp8_quant_max,
    double nvfp4_global_scale_max) {
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2,
        "logits must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        lse.is_cuda() && lse.is_contiguous() &&
            lse.scalar_type() == torch::kFloat,
        "lse must be contiguous CUDA float");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");
    TORCH_CHECK(
        std::isfinite(mxfp8_quant_max) && mxfp8_quant_max > 0.0,
        "MXFP8 quant_max must be finite and positive");
    TORCH_CHECK(
        std::isfinite(nvfp4_global_scale_max) &&
            nvfp4_global_scale_max > 0.0 &&
            nvfp4_global_scale_max <= static_cast<double>(FLT_MAX / 6.0f),
        "NVFP4 global_scale_max must be finite, positive, and representable");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "mixed G-tail quantization requires dimensions divisible by 128");
    TORCH_CHECK(
        lse.numel() == M64 && valid.numel() == M64,
        "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto row_fp8 = torch::empty(
        {M64, N64},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto row_sc = torch::empty(
        {M64 / TILE, N64 / TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {N64, M64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / 64, 512},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_sg = torch::empty(
        {1}, torch::dtype(torch::kFloat32).device(device));

    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_row_nvfp4_col_softmax_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        lse.data_ptr<float>(),
        valid.data_ptr<bool>(),
        reinterpret_cast<uint8_t*>(row_fp8.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        col_sg.data_ptr<float>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(vocab_size),
        static_cast<float>(mxfp8_quant_max),
        static_cast<float>(nvfp4_global_scale_max * 6.0),
        nvfp4_logit_temperature());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(row_fp8, row_sc, col_fp4, col_sc, col_sg);
}

template <bool CONSTANT_SCALE, bool GRAD_CACHE, bool CHUNK_SCALE = false>
void launch_nvfp4_softmax_quant_row_col_tiled_selected(
    const __nv_bfloat16* logits,
    const float* lse,
    const int64_t* targets,
    const bool* valid,
    const float* global_amax,
    float* row_sg_out,
    float* col_sg_out,
    float* row_sg_outer,
    float* col_sg_outer,
    __nv_fp4x2_e2m1* row_fp4,
    uint8_t* row_sc,
    __nv_fp4x2_e2m1* col_fp4,
    uint8_t* col_sc,
    int M,
    int N,
    int vocab_size,
    float global_scale_denom,
    float logit_temperature,
    bool logspace_quant,
    float logspace_affine_a,
    float logspace_affine_b,
    bool row_data_sr,
    bool col_data_sr,
    bool col_zero_sr,
    bool scale_sr,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* rng_state,
    dim3 grid,
    cudaStream_t stream) {
#define LAUNCH_TILED(ROW_DATA, COL_DATA, COL_ZERO, SCALE, LOGSPACE, DIRECT) \
    nvfp4_softmax_quant_row_col_kernel< \
        CONSTANT_SCALE, GRAD_CACHE, CHUNK_SCALE, ROW_DATA, COL_DATA, COL_ZERO, \
        SCALE, LOGSPACE, DIRECT> \
        <<<grid, THREADS, 0, stream>>>( \
            logits, lse, targets, valid, global_amax, row_sg_out, col_sg_out, \
            row_sg_outer, col_sg_outer, row_fp4, row_sc, col_fp4, col_sc, M, N, \
            vocab_size, global_scale_denom, logit_temperature, \
            logspace_affine_a, logspace_affine_b, rng_seed, rng_subsequence_base, \
            rng_state)
    TORCH_CHECK(
        !logspace_quant ||
            (!GRAD_CACHE && !row_data_sr && !col_data_sr && !col_zero_sr &&
             !scale_sr),
        "log-space NVFP4 quantization requires probability target splitting "
        "with deterministic data and scale rounding");
    const bool direct_scale = env_flag(
        "FP4_CCE_V4_NVFP4_G_DIRECT_SCALE", false);
    TORCH_CHECK(
        !direct_scale || (CONSTANT_SCALE && !GRAD_CACHE && !CHUNK_SCALE &&
            !logspace_quant && !row_data_sr && !col_data_sr && !col_zero_sr &&
            !scale_sr),
        "direct NVFP4 scale encoding requires constant global scale, "
        "probability target splitting, and deterministic rounding");
    if (direct_scale) {
        LAUNCH_TILED(false, false, false, false, false, true);
    } else if (logspace_quant) {
        LAUNCH_TILED(false, false, false, false, true, false);
    } else if (row_data_sr && col_data_sr) {
        if (scale_sr) {
            LAUNCH_TILED(true, true, false, true, false, false);
        } else {
            LAUNCH_TILED(true, true, false, false, false, false);
        }
    } else if (row_data_sr && col_zero_sr) {
        if (scale_sr) {
            LAUNCH_TILED(true, false, true, true, false, false);
        } else {
            LAUNCH_TILED(true, false, true, false, false, false);
        }
    } else if (row_data_sr) {
        if (scale_sr) {
            LAUNCH_TILED(true, false, false, true, false, false);
        } else {
            LAUNCH_TILED(true, false, false, false, false, false);
        }
    } else if (col_data_sr) {
        if (scale_sr) {
            LAUNCH_TILED(false, true, false, true, false, false);
        } else {
            LAUNCH_TILED(false, true, false, false, false, false);
        }
    } else if (col_zero_sr) {
        if (scale_sr) {
            LAUNCH_TILED(false, false, true, true, false, false);
        } else {
            LAUNCH_TILED(false, false, true, false, false, false);
        }
    } else if (scale_sr) {
        LAUNCH_TILED(false, false, false, true, false, false);
    } else {
        LAUNCH_TILED(false, false, false, false, false, false);
    }
#undef LAUNCH_TILED
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_softmax_quant_row_col(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    bool constant_scale,
    double global_scale_max) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat, "lse must be contiguous CUDA float");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(lse.numel() == M64 && valid.numel() == M64, "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        std::isfinite(global_scale_max) && global_scale_max > 0.0 &&
            global_scale_max <= static_cast<double>(FLT_MAX / 6.0f),
        "global_scale_max must be finite, positive, and representable");
    const float global_scale_denom = static_cast<float>(global_scale_max * 6.0);
    const float logit_temperature = nvfp4_logit_temperature();
    const bool logspace_quant = env_flag(
        "FP4_CCE_V4_NVFP4_G_LOGSPACE_QUANT", false);
    const bool scale_2d = env_flag(
        "FP4_CCE_V4_NVFP4_G_2D_SCALE", false);
    const bool register_tiled = env_flag(
        "FP4_CCE_V4_NVFP4_G_REGISTER_TILED", false);
    const bool wide_tiled = env_flag(
        "FP4_CCE_V4_NVFP4_G_WIDE_TILED", false);
    const bool row_requant = env_flag(
        "FP4_CCE_V4_NVFP4_G_ROW_REQUANT", false);
    const float logspace_affine_a = env_float(
        "FP4_CCE_V4_NVFP4_G_LOGSPACE_A", 1.61131608f);
    const float logspace_affine_b = env_float(
        "FP4_CCE_V4_NVFP4_G_LOGSPACE_B", 0.93574703f);
    TORCH_CHECK(
        std::isfinite(logspace_affine_a) && std::isfinite(logspace_affine_b),
        "log-space affine coefficients must be finite");

    auto device = logits.device();
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    auto amax = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto extras = nvfp4_cce_extras_from_env();
    const bool any_data_sr = extras.row_data_sr || extras.col_data_sr ||
        extras.col_zero_sr;
    TORCH_CHECK(
        !scale_2d || (constant_scale && !logspace_quant &&
            !any_data_sr && !extras.scale_sr),
        "2D NVFP4 G scaling requires constant global scale, ordinary "
        "probability mapping, and deterministic rounding");
    TORCH_CHECK(
        !register_tiled || (constant_scale && !logspace_quant && !scale_2d &&
            !any_data_sr && !extras.scale_sr),
        "register-tiled NVFP4 G quantization requires constant global scale, "
        "ordinary probability mapping, and deterministic rounding");
    TORCH_CHECK(
        !wide_tiled || (constant_scale && !logspace_quant && !scale_2d &&
            !register_tiled && !any_data_sr && !extras.scale_sr &&
            N64 % WIDE_TILE_COLS == 0),
        "wide-tiled NVFP4 G quantization requires constant global scale, "
        "ordinary probability mapping, deterministic rounding, and N divisible by 256");
    TORCH_CHECK(
        !row_requant || (constant_scale && !logspace_quant && !scale_2d &&
            !register_tiled && !wide_tiled && !any_data_sr &&
            !extras.scale_sr),
        "row-requantized NVFP4 G quantization requires constant global scale, "
        "ordinary probability mapping, and deterministic rounding");
    auto rng_state = maybe_make_cce_advancing_rng_state(logits, extras, stream);
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    if (!constant_scale) {
        amax.zero_();
        const int64_t total = M64 * N64;
        const int blocks = static_cast<int>(std::min<int64_t>((total + THREADS - 1) / THREADS, 4096));
        nvfp4_softmax_amax_kernel<<<blocks, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            valid.data_ptr<bool>(),
            amax.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            logit_temperature);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    const dim3 grid(N64 / TILE, M64 / TILE);
    if (row_requant) {
        const int row_quant_threads = select_staged_row_threads(
            std::getenv("FP4_CCE_V4_NVFP4_G_ROW_THREADS"));
#define LAUNCH_ROW_REQUANT(THREAD_COUNT)                                           \
        nvfp4_softmax_row_quant_lse_kernel<THREAD_COUNT>                          \
            <<<M64, THREAD_COUNT, 0, stream>>>(                                   \
                reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),        \
                lse.data_ptr<float>(),                                             \
                valid.data_ptr<bool>(),                                            \
                sg.data_ptr<float>(),                                              \
                reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),            \
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),                     \
                static_cast<int>(M64),                                             \
                static_cast<int>(N64),                                             \
                static_cast<int>(vocab_size),                                      \
                global_scale_denom,                                                \
                logit_temperature)
        if (row_quant_threads == 1024) {
            LAUNCH_ROW_REQUANT(1024);
        } else if (row_quant_threads == 512) {
            LAUNCH_ROW_REQUANT(512);
        } else if (row_quant_threads == 128) {
            LAUNCH_ROW_REQUANT(128);
        } else {
            LAUNCH_ROW_REQUANT(256);
        }
#undef LAUNCH_ROW_REQUANT
        nvfp4_col_requant_from_row_kernel<false><<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<const uint8_t*>(row_sc.data_ptr()),
            nullptr,
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            global_scale_denom);
    } else if (wide_tiled) {
        const dim3 wide_grid(N64 / WIDE_TILE_COLS, M64 / TILE);
        nvfp4_softmax_quant_row_col_wide_kernel
            <<<wide_grid, WIDE_THREADS, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
                lse.data_ptr<float>(),
                valid.data_ptr<bool>(),
                sg.data_ptr<float>(),
                reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                static_cast<int>(M64),
                static_cast<int>(N64),
                static_cast<int>(vocab_size),
                global_scale_denom,
                logit_temperature);
    } else if (register_tiled) {
        nvfp4_softmax_quant_row_col_register_kernel<<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            valid.data_ptr<bool>(),
            sg.data_ptr<float>(),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            global_scale_denom,
            logit_temperature);
    } else if (scale_2d) {
        nvfp4_softmax_quant_row_col_2d_kernel<<<grid, THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            valid.data_ptr<bool>(),
            sg.data_ptr<float>(),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            global_scale_denom,
            logit_temperature);
    } else if (constant_scale) {
        launch_nvfp4_softmax_quant_row_col_tiled_selected<true, false>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            nullptr,
            valid.data_ptr<bool>(),
            amax.data_ptr<float>(),
            sg.data_ptr<float>(),
            sg.data_ptr<float>(),
            nullptr,
            nullptr,
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            global_scale_denom,
            logit_temperature,
            logspace_quant,
            logspace_affine_a,
            logspace_affine_b,
            extras.row_data_sr,
            extras.col_data_sr,
            extras.col_zero_sr,
            extras.scale_sr,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            grid,
            stream);
    } else {
        launch_nvfp4_softmax_quant_row_col_tiled_selected<false, false>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            nullptr,
            valid.data_ptr<bool>(),
            amax.data_ptr<float>(),
            sg.data_ptr<float>(),
            sg.data_ptr<float>(),
            nullptr,
            nullptr,
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            global_scale_denom,
            logit_temperature,
            logspace_quant,
            logspace_affine_a,
            logspace_affine_b,
            extras.row_data_sr,
            extras.col_data_sr,
            extras.col_zero_sr,
            extras.scale_sr,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            grid,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, sg);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
nvfp4_softmax_quant_row_col_chunk_scaled(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    double global_scale_max) {
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2,
        "logits must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        lse.is_cuda() && lse.is_contiguous() &&
            lse.scalar_type() == torch::kFloat,
        "lse must be contiguous CUDA float");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        M64 % 256 == 0 && N64 % 256 == 0,
        "adaptive block scaling requires M and N to be multiples of 256");
    TORCH_CHECK(
        lse.numel() == M64 && valid.numel() == M64,
        "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        std::isfinite(global_scale_max) && global_scale_max > 0.0 &&
            global_scale_max <= static_cast<double>(FLT_MAX / 6.0f),
        "global_scale_max must be finite, positive, and representable");
    TORCH_CHECK(
        global_scale_max >= 448.0,
        "adaptive block scaling requires global_scale_max >= 448");
    const float global_scale_denom = static_cast<float>(global_scale_max * 6.0);
    const float logit_temperature = nvfp4_logit_temperature();
    const bool logspace_quant = env_flag(
        "FP4_CCE_V4_NVFP4_G_LOGSPACE_QUANT", false);
    const float logspace_affine_a = env_float(
        "FP4_CCE_V4_NVFP4_G_LOGSPACE_A", 1.61131608f);
    const float logspace_affine_b = env_float(
        "FP4_CCE_V4_NVFP4_G_LOGSPACE_B", 0.93574703f);
    TORCH_CHECK(
        std::isfinite(logspace_affine_a) && std::isfinite(logspace_affine_b),
        "log-space affine coefficients must be finite");

    auto device = logits.device();
    auto row_fp4 = torch::empty(
        {M64, N64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M64 / TILE, N64 / 64, 512},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty(
        {1},
        torch::dtype(torch::kFloat32).device(device));
    auto col_fp4 = torch::empty(
        {N64, M64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / 64, 512},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto extras = nvfp4_cce_extras_from_env();
    auto rng_state = maybe_make_cce_advancing_rng_state(logits, extras, stream);
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    const dim3 grid(N64 / TILE, M64 / TILE);
    launch_nvfp4_softmax_quant_row_col_tiled_selected<true, false, true>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        lse.data_ptr<float>(),
        nullptr,
        valid.data_ptr<bool>(),
        nullptr,
        sg.data_ptr<float>(),
        sg.data_ptr<float>(),
        nullptr,
        nullptr,
        reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(vocab_size),
        global_scale_denom,
        logit_temperature,
        logspace_quant,
        logspace_affine_a,
        logspace_affine_b,
        extras.row_data_sr,
        extras.col_data_sr,
        extras.col_zero_sr,
        extras.scale_sr,
        extras.rng_seed,
        extras.rng_subsequence_base,
        rng_state_ptr,
        grid,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        row_fp4, row_sc, sg,
        col_fp4, col_sc, sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_softmax_grad_quant_row_col_tiled(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    double global_scale_max,
    bool block_scale) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat, "lse must be contiguous CUDA float");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(lse.numel() == M64 && targets.numel() == M64 && valid.numel() == M64,
                "lse/target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        !block_scale || (M64 % 256 == 0 && N64 % 256 == 0),
        "adaptive block scaling requires M and N to be multiples of 256");
    TORCH_CHECK(
        std::isfinite(global_scale_max) && global_scale_max > 0.0 &&
            global_scale_max <= static_cast<double>(FLT_MAX / 6.0f),
        "global_scale_max must be finite, positive, and representable");
    TORCH_CHECK(
        !block_scale || global_scale_max >= 448.0,
        "adaptive block scaling requires global_scale_max >= 448");
    const float global_scale_denom = static_cast<float>(global_scale_max * 6.0);
    const float logit_temperature = nvfp4_logit_temperature();

    auto device = logits.device();
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    auto amax = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));

    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto extras = nvfp4_cce_extras_from_env();
    auto rng_state = maybe_make_cce_advancing_rng_state(logits, extras, stream);
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
#define LAUNCH_GRAD_CACHE(BLOCK_SCALE) \
    launch_nvfp4_softmax_quant_row_col_tiled_selected<true, true, BLOCK_SCALE>( \
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), \
        lse.data_ptr<float>(), \
        targets.data_ptr<int64_t>(), \
        valid.data_ptr<bool>(), \
        amax.data_ptr<float>(), \
        sg.data_ptr<float>(), \
        sg.data_ptr<float>(), \
        nullptr, \
        nullptr, \
        reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()), \
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()), \
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()), \
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()), \
        static_cast<int>(M64), \
        static_cast<int>(N64), \
        static_cast<int>(vocab_size), \
        global_scale_denom, \
        logit_temperature, \
        false, \
        1.61131608f, \
        0.93574703f, \
        extras.row_data_sr, \
        extras.col_data_sr, \
        extras.col_zero_sr, \
        extras.scale_sr, \
        extras.rng_seed, \
        extras.rng_subsequence_base, \
        rng_state_ptr, \
        grid, \
        stream)
    if (block_scale) {
        LAUNCH_GRAD_CACHE(true);
    } else {
        LAUNCH_GRAD_CACHE(false);
    }
#undef LAUNCH_GRAD_CACHE
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_softmax_quant_row_col_tma(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat, "lse must be contiguous CUDA float");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(lse.numel() == M64 && valid.numel() == M64, "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M64, N64, tk_v3::V3_BUFF_DIM_Y, tk_v3::V3_BUFF_DIM_X, N64, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), N64, M64, tk_v3::V3_BUFF_DIM_X, tk_v3::V3_BUFF_DIM_Y, M64, 4);

    const int64_t ntm_r = M64 / 128;
    const int64_t ntk_r = N64 / 64;
    const int64_t ntm_c = N64 / 128;
    const int64_t ntk_c = M64 / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    const int shmem = tk_v3::v3_shmem_size<true>();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaFuncSetAttribute(
        nvfp4_softmax_quant_row_col_tma_kernel<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem);

    const dim3 grid(N64 / TILE, M64 / TILE);
    nvfp4_softmax_quant_row_col_tma_kernel<false><<<grid, tk_v3::V3_THREADS, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        lse.data_ptr<float>(),
        nullptr,
        valid.data_ptr<bool>(),
        tmap_out,
        tmap_out_t,
        tmap_sc_row,
        tmap_sc_col,
        sg.data_ptr<float>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(vocab_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_softmax_grad_quant_row_col_tma(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat, "lse must be contiguous CUDA float");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(lse.numel() == M64 && targets.numel() == M64 && valid.numel() == M64,
                "lse/target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));

    alignas(64) CUtensorMap tmap_out{}, tmap_out_t{}, tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M64, N64, tk_v3::V3_BUFF_DIM_Y, tk_v3::V3_BUFF_DIM_X, N64, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), N64, M64, tk_v3::V3_BUFF_DIM_X, tk_v3::V3_BUFF_DIM_Y, M64, 4);

    const int64_t ntm_r = M64 / 128;
    const int64_t ntk_r = N64 / 64;
    const int64_t ntm_c = N64 / 128;
    const int64_t ntk_c = M64 / 64;
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    const int shmem = tk_v3::v3_shmem_size<true>();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    cudaFuncSetAttribute(
        nvfp4_softmax_quant_row_col_tma_kernel<true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem);

    const dim3 grid(N64 / TILE, M64 / TILE);
    nvfp4_softmax_quant_row_col_tma_kernel<true><<<grid, tk_v3::V3_THREADS, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        lse.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        tmap_out,
        tmap_out_t,
        tmap_sc_row,
        tmap_sc_col,
        sg.data_ptr<float>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(vocab_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_softmax_quant_row_col_staged(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int row_threads = select_staged_row_threads(std::getenv("FP4_CCE_V4_NVFP4_STAGED_ROW_THREADS"));
    const float s_enc = NVFP4_GLOBAL_SCALE_DENOM;
    const float sg_value = 1.0f / NVFP4_GLOBAL_SCALE_DENOM;
    const auto extras = nvfp4_cce_extras_from_env();
    auto rng_state = maybe_make_cce_advancing_rng_state(logits, extras, stream);
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    if (row_threads == 1024) {
        launch_nvfp4_softmax_row_quant_staged_selected<1024>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            sg.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            s_enc,
            sg_value,
            extras.data_sr,
            extras.scale_sr,
            extras.with_rht,
            extras.random_sign_mask,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            stream);
    } else if (row_threads == 512) {
        launch_nvfp4_softmax_row_quant_staged_selected<512>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            sg.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            s_enc,
            sg_value,
            extras.data_sr,
            extras.scale_sr,
            extras.with_rht,
            extras.random_sign_mask,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            stream);
    } else {
        launch_nvfp4_softmax_row_quant_staged_selected<ROW_THREADS>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            sg.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            s_enc,
            sg_value,
            extras.data_sr,
            extras.scale_sr,
            extras.with_rht,
            extras.random_sign_mask,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid(N64 / TILE, M64 / TILE);
    launch_nvfp4_col_quant_from_probs_selected(
        reinterpret_cast<const __nv_bfloat16*>(probs.data_ptr()),
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        static_cast<int>(M64),
        static_cast<int>(N64),
        s_enc,
        extras.data_sr,
        extras.scale_sr,
        extras.with_rht,
        extras.random_sign_mask,
        extras.rng_seed,
        extras.rng_subsequence_base,
        rng_state_ptr,
        grid,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(loss.reshape({}), row_fp4, row_sc, col_fp4, col_sc, sg);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_softmax_grad_quant_row_col_staged(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto grad_probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / 64, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int row_threads = select_staged_row_threads(std::getenv("FP4_CCE_V4_NVFP4_STAGED_ROW_THREADS"));
    const float s_enc = NVFP4_GLOBAL_SCALE_DENOM;
    const float sg_value = 1.0f / NVFP4_GLOBAL_SCALE_DENOM;
    const auto extras = nvfp4_cce_extras_from_env();
    auto rng_state = maybe_make_cce_advancing_rng_state(logits, extras, stream);
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    if (row_threads == 1024) {
        launch_nvfp4_softmax_row_quant_staged_selected<1024>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            sg.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            s_enc,
            sg_value,
            extras.data_sr,
            extras.scale_sr,
            extras.with_rht,
            extras.random_sign_mask,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            stream);
    } else if (row_threads == 512) {
        launch_nvfp4_softmax_row_quant_staged_selected<512>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            sg.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            s_enc,
            sg_value,
            extras.data_sr,
            extras.scale_sr,
            extras.with_rht,
            extras.random_sign_mask,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            stream);
    } else {
        launch_nvfp4_softmax_row_quant_staged_selected<ROW_THREADS>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            sg.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            s_enc,
            sg_value,
            extras.data_sr,
            extras.scale_sr,
            extras.with_rht,
            extras.random_sign_mask,
            extras.rng_seed,
            extras.rng_subsequence_base,
            rng_state_ptr,
            stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    dim3 grid(N64 / TILE, M64 / TILE);
    launch_nvfp4_col_quant_from_probs_selected(
        reinterpret_cast<const __nv_bfloat16*>(grad_probs.data_ptr()),
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        static_cast<int>(M64),
        static_cast<int>(N64),
        s_enc,
        extras.data_sr,
        extras.scale_sr,
        extras.with_rht,
        extras.random_sign_mask,
        extras.rng_seed,
        extras.rng_subsequence_base,
        rng_state_ptr,
        grid,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(loss.reshape({}), row_fp4, row_sc, col_fp4, col_sc, sg);
}

std::tuple<torch::Tensor, torch::Tensor>
nvfp4_col_requant_from_row(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor row_normalization) {
    TORCH_CHECK(
        row_fp4.is_cuda() && row_fp4.is_contiguous() && row_fp4.dim() == 2 &&
            row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2,
        "row_fp4 must be contiguous CUDA float4_e2m1fn_x2 [M, N/2]");
    TORCH_CHECK(
        row_sc.is_cuda() && row_sc.is_contiguous() && row_sc.dim() == 3 &&
            row_sc.scalar_type() == torch::kFloat8_e4m3fn,
        "row_sc must be contiguous CUDA float8_e4m3fn");
    TORCH_CHECK(
        row_fp4.device() == row_sc.device(),
        "row FP4 values and scales must be on one CUDA device");
    TORCH_CHECK(
        row_normalization.is_cuda() && row_normalization.is_contiguous() &&
            row_normalization.scalar_type() == torch::kFloat32 &&
            row_normalization.dim() == 1 &&
            row_normalization.device() == row_fp4.device(),
        "row_normalization must be contiguous CUDA float32 [M]");

    const int64_t M64 = row_fp4.size(0);
    const int64_t N64 = row_fp4.size(1) * 2;
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "row-requantized dimensions must be multiples of 128");
    TORCH_CHECK(
        row_sc.size(0) == M64 / TILE && row_sc.size(1) == N64 / 64 &&
            row_sc.size(2) == 512,
        "row scale shape mismatch");
    TORCH_CHECK(
        row_normalization.numel() == M64,
        "row normalization length mismatch");

    auto device = row_fp4.device();
    auto col_fp4 = torch::empty(
        {N64, M64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / 64, 512},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto extras = nvfp4_cce_extras_from_env();
    auto rng_state = (extras.col_data_sr || extras.col_zero_sr)
        ? make_cce_advancing_rng_state(
              row_fp4,
              extras.rng_seed,
              extras.rng_subsequence_base,
              stream)
        : torch::Tensor();
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    const dim3 grid(N64 / TILE, M64 / TILE);
#define LAUNCH_NVFP4_COL_REQUANT(COL_SR, ZERO_SR)                              \
    nvfp4_col_requant_from_row_kernel<true, true, COL_SR, ZERO_SR>              \
        <<<grid, THREADS, 0, stream>>>(                                         \
            reinterpret_cast<const __nv_fp4x2_e2m1*>(row_fp4.data_ptr()),       \
            reinterpret_cast<const uint8_t*>(row_sc.data_ptr()),                \
            row_normalization.data_ptr<float>(),                                \
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),             \
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),                      \
            static_cast<int>(M64), static_cast<int>(N64),                       \
            NVFP4_GLOBAL_SCALE_DENOM,                                           \
            extras.rng_seed,                                                    \
            extras.rng_subsequence_base,                                        \
            rng_state_ptr)
    if (extras.col_data_sr) {
        LAUNCH_NVFP4_COL_REQUANT(true, false);
    } else if (extras.col_zero_sr) {
        LAUNCH_NVFP4_COL_REQUANT(false, true);
    } else {
        LAUNCH_NVFP4_COL_REQUANT(false, false);
    }
#undef LAUNCH_NVFP4_COL_REQUANT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
nvfp4_col_requant_from_mxfp8_row(
    torch::Tensor row_fp8,
    torch::Tensor row_sc,
    torch::Tensor row_normalization,
    double global_scale_max) {
    TORCH_CHECK(
        row_fp8.is_cuda() && row_fp8.is_contiguous() && row_fp8.dim() == 2 &&
            row_fp8.scalar_type() == torch::kFloat8_e4m3fn,
        "row_fp8 must be contiguous CUDA float8_e4m3fn [M, N]");
    TORCH_CHECK(
        row_sc.is_cuda() && row_sc.is_contiguous() && row_sc.dim() == 4 &&
            row_sc.scalar_type() == torch::kUInt8,
        "row_sc must be contiguous CUDA uint8 MXFP8 scales");
    TORCH_CHECK(
        row_fp8.device() == row_sc.device(),
        "row MXFP8 values and scales must be on one CUDA device");
    TORCH_CHECK(
        row_normalization.is_cuda() && row_normalization.is_contiguous() &&
            row_normalization.scalar_type() == torch::kFloat32 &&
            row_normalization.dim() == 1 &&
            row_normalization.device() == row_fp8.device(),
        "row_normalization must be contiguous CUDA float32 [M]");
    TORCH_CHECK(
        std::isfinite(global_scale_max) && global_scale_max > 0.0 &&
            global_scale_max <= static_cast<double>(FLT_MAX / 6.0f),
        "global_scale_max must be finite, positive, and representable");

    const int64_t M64 = row_fp8.size(0);
    const int64_t N64 = row_fp8.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "row-requantized dimensions must be multiples of 128");
    TORCH_CHECK(
        row_sc.size(0) == M64 / TILE && row_sc.size(1) == N64 / TILE &&
            row_sc.size(2) == 32 && row_sc.size(3) == 16,
        "MXFP8 row scale shape mismatch");
    TORCH_CHECK(
        row_normalization.numel() == M64,
        "row normalization length mismatch");

    auto device = row_fp8.device();
    auto col_fp4 = torch::empty(
        {N64, M64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / 64, 512},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_sg = torch::empty(
        {1}, torch::dtype(torch::kFloat32).device(device));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto extras = nvfp4_cce_extras_from_env();
    auto rng_state = (extras.col_data_sr || extras.col_zero_sr)
        ? make_cce_advancing_rng_state(
              row_fp8,
              extras.rng_seed,
              extras.rng_subsequence_base,
              stream)
        : torch::Tensor();
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    const dim3 grid(N64 / TILE, M64 / TILE);
#define LAUNCH_MXFP8_NVFP4_COL_REQUANT(COL_SR, ZERO_SR)                      \
    nvfp4_col_requant_from_mxfp8_row_kernel<COL_SR, ZERO_SR>                \
        <<<grid, THREADS, 0, stream>>>(                                      \
            reinterpret_cast<const uint8_t*>(row_fp8.data_ptr()),           \
            reinterpret_cast<const uint8_t*>(row_sc.data_ptr()),            \
            row_normalization.data_ptr<float>(),                             \
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),         \
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),                   \
            col_sg.data_ptr<float>(),                                        \
            static_cast<int>(M64), static_cast<int>(N64),                    \
            static_cast<float>(global_scale_max * 6.0),                      \
            extras.rng_seed,                                                 \
            extras.rng_subsequence_base,                                     \
            rng_state_ptr)
    if (extras.col_data_sr) {
        LAUNCH_MXFP8_NVFP4_COL_REQUANT(true, false);
    } else if (extras.col_zero_sr) {
        LAUNCH_MXFP8_NVFP4_COL_REQUANT(false, true);
    } else {
        LAUNCH_MXFP8_NVFP4_COL_REQUANT(false, false);
    }
#undef LAUNCH_MXFP8_NVFP4_COL_REQUANT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(col_fp4, col_sc, col_sg);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp8_col_requant_from_mxfp8_row(
    torch::Tensor row_fp8,
    torch::Tensor row_sc,
    torch::Tensor row_normalization,
    double quant_max) {
    TORCH_CHECK(
        row_fp8.is_cuda() && row_fp8.is_contiguous() && row_fp8.dim() == 2 &&
            row_fp8.scalar_type() == torch::kFloat8_e4m3fn,
        "row_fp8 must be contiguous CUDA float8_e4m3fn [M, N]");
    TORCH_CHECK(
        row_sc.is_cuda() && row_sc.is_contiguous() && row_sc.dim() == 4 &&
            row_sc.scalar_type() == torch::kUInt8,
        "row_sc must be contiguous CUDA uint8 MXFP8 scales");
    TORCH_CHECK(
        row_fp8.device() == row_sc.device(),
        "row MXFP8 values and scales must be on one CUDA device");
    TORCH_CHECK(
        row_normalization.is_cuda() && row_normalization.is_contiguous() &&
            row_normalization.scalar_type() == torch::kFloat32 &&
            row_normalization.dim() == 1 &&
            row_normalization.device() == row_fp8.device(),
        "row_normalization must be contiguous CUDA float32 [M]");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    const int64_t M64 = row_fp8.size(0);
    const int64_t N64 = row_fp8.size(1);
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "row-requantized dimensions must be multiples of 128");
    TORCH_CHECK(
        row_sc.size(0) == M64 / TILE && row_sc.size(1) == N64 / TILE &&
            row_sc.size(2) == 32 && row_sc.size(3) == 16,
        "MXFP8 row scale shape mismatch");
    TORCH_CHECK(
        row_normalization.numel() == M64,
        "row normalization length mismatch");

    auto device = row_fp8.device();
    auto col_fp8 = torch::empty(
        {N64, M64},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    const dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp8_col_requant_from_mxfp8_row_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(row_fp8.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        row_normalization.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<float>(quant_max));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(col_fp8, col_sc);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp8_quant_row", &mxfp8_quant_row,
          "Quantize BF16 input directly into the row MXFP8 operand",
          pybind11::arg("input"), pybind11::arg("quant_max") = 448.0);
    m.def("mxfp8_quant_row_col", &mxfp8_quant_row_col,
          "Quantize BF16 input directly into row/transpose MXFP8 operands");
    m.def("mxfp8_quant_col", &mxfp8_quant_col,
          "Quantize BF16 input into a transposed MXFP8 operand",
          pybind11::arg("input"), pybind11::arg("quant_max") = 448.0);
    m.def(
          "mxfp4_quant_row_mxfp8_col",
          &mxfp4_quant_row_mxfp8_col,
          "Quantize BF16 input into MXFP4 rows and MXFP8 transpose rows");
    m.def("mxfp8_rmsnorm_quant_row_col", &mxfp8_rmsnorm_quant_row_col,
          "Fuse RMSNorm with row/transpose MXFP8 operand production");
    m.def("mxfp8_softmax_quant_row_col", &mxfp8_softmax_quant_row_col,
          "Compute a softmax tail and directly emit row/transpose MXFP8 operands");
    m.def(
          "mxfp8_row_nvfp4_col_softmax_quant",
          &mxfp8_row_nvfp4_col_softmax_quant,
          "Compute a softmax tail and emit MXFP8 rows plus NVFP4 transpose rows",
          pybind11::arg("logits"), pybind11::arg("lse"),
          pybind11::arg("valid"), pybind11::arg("vocab_size"),
          pybind11::arg("mxfp8_quant_max") = 448.0,
          pybind11::arg("nvfp4_global_scale_max") = 448.0);
    m.def("nvfp4_softmax_quant_row_col", &nvfp4_softmax_quant_row_col,
          "Compute softmax(logits) from precomputed LSE and directly emit row/col NVFP4 P caches",
          pybind11::arg("logits"), pybind11::arg("lse"), pybind11::arg("valid"),
          pybind11::arg("vocab_size"), pybind11::arg("constant_scale"),
          pybind11::arg("global_scale_max") = 448.0);
    m.def(
          "nvfp4_col_requant_from_row",
          &nvfp4_col_requant_from_row,
          "Build the transposed NVFP4 operand from a normalized row operand");
    m.def(
          "nvfp4_col_requant_from_mxfp8_row",
          &nvfp4_col_requant_from_mxfp8_row,
          "Build a transposed NVFP4 operand from an MXFP8 numerator row",
          pybind11::arg("row_fp8"), pybind11::arg("row_sc"),
          pybind11::arg("row_normalization"),
          pybind11::arg("global_scale_max") = 448.0);
    m.def(
          "mxfp8_col_requant_from_mxfp8_row",
          &mxfp8_col_requant_from_mxfp8_row,
          "Build a transposed MXFP8 operand from an MXFP8 numerator row",
          pybind11::arg("row_fp8"), pybind11::arg("row_sc"),
          pybind11::arg("row_normalization"),
          pybind11::arg("quant_max") = 448.0);
    m.def(
          "nvfp4_softmax_quant_row_col_chunk_scaled",
          &nvfp4_softmax_quant_row_col_chunk_scaled,
          "Emit row/col NVFP4 P caches with adaptive per-block prescaling",
          pybind11::arg("logits"), pybind11::arg("lse"),
          pybind11::arg("valid"), pybind11::arg("vocab_size"),
          pybind11::arg("global_scale_max") = 131072.0);
    m.def("nvfp4_softmax_grad_quant_row_col_tiled", &nvfp4_softmax_grad_quant_row_col_tiled,
          "Compute G=(softmax(logits)-onehot) from precomputed LSE and directly emit row/col NVFP4 G caches",
          pybind11::arg("logits"), pybind11::arg("lse"), pybind11::arg("targets"),
          pybind11::arg("valid"), pybind11::arg("vocab_size"),
          pybind11::arg("global_scale_max") = 448.0,
          pybind11::arg("block_scale") = false);
    m.def("nvfp4_softmax_quant_row_col_tma", &nvfp4_softmax_quant_row_col_tma,
          "Compute softmax(logits) from precomputed LSE and emit row/col NVFP4 P caches with v5 TMA stores");
    m.def("nvfp4_softmax_grad_quant_row_col_tma", &nvfp4_softmax_grad_quant_row_col_tma,
          "Compute G=(softmax(logits)-onehot) from precomputed LSE and emit row/col NVFP4 G caches with v5 TMA stores");
    m.def("nvfp4_softmax_quant_row_col_staged", &nvfp4_softmax_quant_row_col_staged,
          "Compute softmax/loss plus row NVFP4, then col NVFP4 from staged BF16 probabilities");
    m.def("nvfp4_softmax_grad_quant_row_col_staged", &nvfp4_softmax_grad_quant_row_col_staged,
          "Compute softmax/loss and directly emit row/col NVFP4 G=(P-onehot) caches");
}
