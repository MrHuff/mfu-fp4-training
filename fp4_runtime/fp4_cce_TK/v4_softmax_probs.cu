#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <type_traits>

#define TK_STANDALONE
#include "../TK_quantisation/nvfp4_v5/core.cuh"

namespace py = pybind11;

namespace {

constexpr int ROW_THREADS = 256;
constexpr int MAX_TOPK_SPLIT = 4;
constexpr int NVFP4_BLOCK = 16;
constexpr int NVFP4_TILE = 128;
constexpr int MXFP4_BLOCK = 32;
constexpr int FUSED_ROW_NONE = 0;
constexpr int FUSED_ROW_NVFP4 = 1;
constexpr int FUSED_ROW_MXFP4 = 2;
constexpr int FUSED_ROW_MXFP8 = 3;
constexpr int MXFP4_TAIL_NATIVE = 0;
constexpr int MXFP4_TAIL_POLY4 = 1;
constexpr int MXFP4_TAIL_POLY4_REPRESENTED = 2;
constexpr int MXFP4_TAIL_AFFINE_REPRESENTED = 3;
constexpr int MXFP4_TAIL_HYBRID25 = 4;
constexpr int MXFP4_TAIL_HYBRID25_REPRESENTED = 5;
constexpr int MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED = 6;

constexpr unsigned long long FUSED_ROW_SR_INVOCATION_STRIDE = 1ull << 40;
__device__ unsigned long long fused_row_sr_invocation_offset = 0;

__global__ void prepare_fused_row_rng_state_kernel(
    unsigned long long* rng_state,
    unsigned long long rng_seed,
    unsigned long long rng_subsequence_base,
    unsigned long long* persistent_rng_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (persistent_rng_state != nullptr) {
            const unsigned long long subsequence = atomicAdd(
                persistent_rng_state + 1,
                FUSED_ROW_SR_INVOCATION_STRIDE);
            rng_state[0] = persistent_rng_state[0] ^ subsequence;
        } else {
            const unsigned long long offset = atomicAdd(
                &fused_row_sr_invocation_offset,
                FUSED_ROW_SR_INVOCATION_STRIDE);
            rng_state[0] = rng_seed ^ (rng_subsequence_base + offset);
        }
    }
}

struct alignas(16) NVFP4BFloat16x8 {
    __nv_bfloat162 values[4];
};

struct alignas(16) FP8x16 {
    uint8_t values[16];
};

float mxfp8_logit_temperature() {
    const char* value = std::getenv("FP4_CCE_V4_LOGIT_TEMPERATURE");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("FP4_CCE_V4_MXFP8_LOGIT_TEMPERATURE");
    }
    if (value == nullptr || value[0] == '\0') return 1.0f;
    char* end = nullptr;
    const float temperature = std::strtof(value, &end);
    TORCH_CHECK(
        end != value && end != nullptr && end[0] == '\0' &&
            std::isfinite(temperature) &&
            temperature >= 0.5f && temperature <= 2.0f,
        "FP4_CCE_V4_LOGIT_TEMPERATURE must be in [0.5, 2]");
    return temperature;
}

bool env_flag(const char* name, bool default_value = false) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return !(value[0] == '\0' || value[0] == '0' || value[0] == 'f' ||
             value[0] == 'F' || value[0] == 'n' || value[0] == 'N');
}

uint64_t env_u64(const char* name, uint64_t default_value = 0) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
}

torch::Tensor make_fused_row_rng_state(
    const torch::Tensor& input,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream,
    const torch::Tensor& persistent_rng_state = torch::Tensor()) {
    if (persistent_rng_state.defined()) {
        TORCH_CHECK(
            persistent_rng_state.is_cuda() && persistent_rng_state.is_contiguous() &&
                persistent_rng_state.scalar_type() == torch::kInt64 &&
                persistent_rng_state.dim() == 1 && persistent_rng_state.numel() == 2,
            "persistent fused-row SR state must be contiguous CUDA int64[2]");
        TORCH_CHECK(
            persistent_rng_state.device() == input.device(),
            "persistent fused-row SR state must be on the input device");
    }
    auto rng_state = torch::empty(
        {1}, torch::dtype(torch::kInt64).device(input.device()));
    prepare_fused_row_rng_state_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
        static_cast<unsigned long long>(rng_seed),
        static_cast<unsigned long long>(rng_subsequence_base),
        persistent_rng_state.defined()
            ? reinterpret_cast<unsigned long long*>(
                  persistent_rng_state.data_ptr<int64_t>())
            : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rng_state;
}

int mxfp4_tail_math_mode() {
    const char* value = std::getenv("FP4_CCE_V4_MXFP4_G_TAIL_MATH");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "native") == 0) {
        return MXFP4_TAIL_NATIVE;
    }
    if (std::strcmp(value, "poly4") == 0) return MXFP4_TAIL_POLY4;
    if (std::strcmp(value, "poly4_represented") == 0) {
        return MXFP4_TAIL_POLY4_REPRESENTED;
    }
    if (std::strcmp(value, "affine_represented") == 0) {
        return MXFP4_TAIL_AFFINE_REPRESENTED;
    }
    if (std::strcmp(value, "hybrid25") == 0) return MXFP4_TAIL_HYBRID25;
    if (std::strcmp(value, "hybrid25_represented") == 0) {
        return MXFP4_TAIL_HYBRID25_REPRESENTED;
    }
    if (std::strcmp(value, "logcode_exact_represented") == 0) {
        return MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED;
    }
    TORCH_CHECK(
        false,
        "FP4_CCE_V4_MXFP4_G_TAIL_MATH must be native, poly4, "
        "poly4_represented, affine_represented, hybrid25, or "
        "hybrid25_represented, or logcode_exact_represented");
    return MXFP4_TAIL_NATIVE;
}

float mxfp4_single_pass_anchor() {
    const char* value = std::getenv("FP4_CCE_V4_MXFP4_G_FIXED_ANCHOR");
    if (value == nullptr || value[0] == '\0') return 32.0f;
    char* end = nullptr;
    const float anchor = std::strtof(value, &end);
    TORCH_CHECK(
        end != value && end != nullptr && end[0] == '\0' &&
            std::isfinite(anchor) && anchor >= -64.0f && anchor <= 64.0f,
        "FP4_CCE_V4_MXFP4_G_FIXED_ANCHOR must be finite and in [-64, 64]");
    return anchor;
}

__device__ __forceinline__ uint8_t fp8_byte(__nv_fp8_e4m3 value) {
    return *reinterpret_cast<uint8_t*>(&value);
}

__device__ __forceinline__ float fp8_from_byte(uint8_t raw) {
    __nv_fp8_e4m3 value;
    *reinterpret_cast<uint8_t*>(&value) = raw;
    return static_cast<float>(value);
}

__device__ __forceinline__ float logit_value(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float logit_value(__nv_fp8_e4m3 value) {
    return static_cast<float>(value);
}

template <bool CENTERED_LOGITS, typename LOGIT_T>
__device__ __forceinline__ float decoded_logit_value(
    LOGIT_T value,
    const __nv_bfloat16* __restrict__ logit_centers,
    int row,
    int col,
    int center_stride) {
    float decoded = logit_value(value);
    if constexpr (CENTERED_LOGITS) {
        static_assert(std::is_same_v<LOGIT_T, __nv_fp8_e4m3>);
        decoded += __bfloat162float(
            logit_centers[static_cast<int64_t>(row) * center_stride + col / 32]);
    }
    return decoded;
}

__device__ __forceinline__ void remove_logit(__nv_bfloat16* value) {
    *value = __float2bfloat16(-INFINITY);
}

__device__ __forceinline__ void remove_logit(__nv_fp8_e4m3* value) {
    // E4M3FN has no infinity. Its finite minimum is far enough below the row
    // maximum that the selected probability numerator rounds to zero.
    *value = static_cast<__nv_fp8_e4m3>(-448.0f);
}

__device__ __forceinline__ void clear_mxfp4_row_value(
    __nv_fp4x2_e2m1* row_fp4,
    int row,
    int col,
    int N) {
    const int64_t byte_offset =
        static_cast<int64_t>(row) * (N / 2) + col / 2;
    auto* bytes = reinterpret_cast<uint8_t*>(row_fp4);
    auto* word = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(bytes + byte_offset) &
        ~static_cast<uintptr_t>(3));
    const int shift =
        static_cast<int>(byte_offset & 3) * 8 + (col & 1) * 4;
    atomicAnd(word, ~(0x0fu << shift));
}

__device__ __forceinline__ uint32_t fused_sr_bits(uint32_t key) {
    key ^= key >> 16;
    key *= 0x7feb352du;
    key ^= key >> 15;
    key *= 0x846ca68bu;
    key ^= key >> 16;
    return key;
}

__device__ __forceinline__ uint32_t fused_sr_next(uint32_t value) {
    // Full-period modulo 2^32; the block mixer supplies independent starts.
    return value * 747796405u + 2891336453u;
}

__device__ __forceinline__ float exp_poly4_range_reduced(float x) {
    // 2^r on r in [-0.5, 0.5], with a nearest-integer base-2 range
    // reduction. The degree-4 truncation has a worst-case relative error of
    // roughly 5e-5 on this interval and moves work from SFUs to FMA pipes.
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float C1 = 0.6931471805599453f;
    constexpr float C2 = 0.2402265069591007f;
    constexpr float C3 = 0.05550410866482158f;
    constexpr float C4 = 0.009618129107628477f;
    const float y = x * LOG2E;
    const int exponent = __float2int_rn(y);
    if (exponent <= -127) return 0.0f;
    if (exponent >= 128) return INFINITY;
    const float r = y - static_cast<float>(exponent);
    float polynomial = fmaf(C4, r, C3);
    polynomial = fmaf(polynomial, r, C2);
    polynomial = fmaf(polynomial, r, C1);
    polynomial = fmaf(polynomial, r, 1.0f);
    const float scale = __int_as_float((exponent + 127) << 23);
    return polynomial * scale;
}

template <int TAIL_MATH>
__device__ __forceinline__ float mxfp4_tail_exp(float x, int element) {
    if constexpr (
        TAIL_MATH == MXFP4_TAIL_NATIVE ||
        TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED) {
        return __expf(x);
    } else if constexpr (
        TAIL_MATH == MXFP4_TAIL_HYBRID25 ||
        TAIL_MATH == MXFP4_TAIL_HYBRID25_REPRESENTED) {
        return (element & 3) == 0 ? __expf(x) : exp_poly4_range_reduced(x);
    } else {
        return exp_poly4_range_reduced(x);
    }
}

// Accumulate twice the represented positive E2M1 values from eight packed
// nibbles. PRMT performs the tiny code-to-weight lookup and DP4A sums four
// lanes at once, avoiding a long switch/select chain in the tail producer.
__device__ __forceinline__ int mxfp4_qsum_x2_from_word(
    uint32_t word,
    int accumulator) {
    const uint32_t low_codes = word & 0x07070707u;
    const uint32_t high_codes = (word >> 4) & 0x07070707u;
    const uint32_t low_selectors =
        (low_codes & 0x0fu) |
        ((low_codes >> 4) & 0x00f0u) |
        ((low_codes >> 8) & 0x0f00u) |
        ((low_codes >> 12) & 0xf000u);
    const uint32_t high_selectors =
        (high_codes & 0x0fu) |
        ((high_codes >> 4) & 0x00f0u) |
        ((high_codes >> 8) & 0x0f00u) |
        ((high_codes >> 12) & 0xf000u);
    uint32_t low_weights;
    uint32_t high_weights;
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(low_weights)
        : "r"(0x03020100u), "r"(0x0c080604u), "r"(low_selectors));
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(high_weights)
        : "r"(0x03020100u), "r"(0x0c080604u), "r"(high_selectors));
    asm("dp4a.s32.s32 %0, %1, %2, %0;"
        : "+r"(accumulator)
        : "r"(low_weights), "r"(0x01010101u));
    asm("dp4a.s32.s32 %0, %1, %2, %0;"
        : "+r"(accumulator)
        : "r"(high_weights), "r"(0x01010101u));
    return accumulator;
}

__device__ __forceinline__ float nvfp4_quant_coeff(
    float block_amax,
    float global_scale_denom,
    uint8_t* scale_byte) {
    __nv_fp8_e4m3 multiplier;
    if (block_amax <= 1.0e-9f) {
        multiplier = static_cast<__nv_fp8_e4m3>(448.0f);
    } else {
        multiplier = static_cast<__nv_fp8_e4m3>(
            fminf(6.0f / (block_amax * global_scale_denom), FLT_MAX));
    }
    const float multiplier_f = static_cast<float>(multiplier);
    const float inverse = multiplier_f > 0.0f ? 1.0f / multiplier_f : 448.0f;
    *scale_byte = fp8_byte(static_cast<__nv_fp8_e4m3>(inverse));
    return multiplier_f * global_scale_denom;
}

__device__ __forceinline__ int nvfp4_scale_offset(
    int row,
    int block,
    int blocks_per_64) {
    const int tile_row = row / NVFP4_TILE;
    const int row_in_tile = row & (NVFP4_TILE - 1);
    const int tile_col = block / (NVFP4_TILE / NVFP4_BLOCK);
    const int scale_index = block & ((NVFP4_TILE / NVFP4_BLOCK) - 1);
    const int lane = row_in_tile & 31;
    const int group = row_in_tile >> 5;
    const int base =
        (tile_row * blocks_per_64 + tile_col * 2) * 512 +
        lane * 16 + group * 4;
    return base + (scale_index >> 2) * 512 + (scale_index & 3);
}

template <bool DATA_SR>
__device__ __forceinline__ float quantize_nvfp4_exp_block(
    __nv_bfloat16* __restrict__ logits,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    int row,
    int block,
    int N,
    int vocab_size,
    float row_max,
    float logit_temperature,
    float global_scale_denom,
    uint32_t sr_nonce) {
    const int col0 = block * NVFP4_BLOCK;
    const int64_t row_offset = static_cast<int64_t>(row) * N;
    float values[NVFP4_BLOCK];
    float block_amax = 0.0f;
    float block_sum = 0.0f;
    if (col0 + NVFP4_BLOCK <= vocab_size) {
        const auto* packed_logits =
            reinterpret_cast<const NVFP4BFloat16x8*>(
                logits + row_offset + col0);
        const NVFP4BFloat16x8 first = packed_logits[0];
        const NVFP4BFloat16x8 second = packed_logits[1];
        #pragma unroll
        for (int pair = 0; pair < NVFP4_BLOCK / 2; ++pair) {
            const __nv_bfloat162 packed = pair < 4
                ? first.values[pair]
                : second.values[pair - 4];
            const float2 raw = __bfloat1622float2(packed);
            const float numerator0 = __expf(
                fmaf(logit_temperature, raw.x, -row_max));
            const float numerator1 = __expf(
                fmaf(logit_temperature, raw.y, -row_max));
            block_sum += numerator0 + numerator1;
            const float rounded0 = __bfloat162float(
                __float2bfloat16(numerator0));
            const float rounded1 = __bfloat162float(
                __float2bfloat16(numerator1));
            values[pair * 2] = rounded0;
            values[pair * 2 + 1] = rounded1;
            block_amax = fmaxf(block_amax, fmaxf(rounded0, rounded1));
        }
    } else {
        #pragma unroll
        for (int element = 0; element < NVFP4_BLOCK; ++element) {
            const int col = col0 + element;
            float probability_numerator = 0.0f;
            if (col < vocab_size) {
                const float z = logit_temperature *
                    __bfloat162float(logits[row_offset + col]);
                probability_numerator = __expf(z - row_max);
                block_sum += probability_numerator;
            }
            const float rounded = __bfloat162float(
                __float2bfloat16(probability_numerator));
            values[element] = rounded;
            block_amax = fmaxf(block_amax, rounded);
        }
    }

    uint8_t scale_byte = 0;
    const float coefficient = nvfp4_quant_coeff(
        block_amax, global_scale_denom, &scale_byte);
    row_sc[nvfp4_scale_offset(row, block, N / 64)] = scale_byte;
    const int64_t packed_offset =
        static_cast<int64_t>(row) * (N / 2) + col0 / 2;
    uint64_t packed_values = 0;
    if constexpr (DATA_SR) {
        const uint32_t block_key =
            static_cast<uint32_t>(row) * (N / NVFP4_BLOCK) + block;
        uint32_t random_bits = fused_sr_bits(sr_nonce + block_key);
        #pragma unroll
        for (int element = 0; element < NVFP4_BLOCK; element += 4) {
            const float2 in01 = {values[element], values[element + 1]};
            const float2 in23 = {values[element + 2], values[element + 3]};
            const float2 scale = {coefficient, coefficient};
            const auto packed4 =
                transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                    in01, in23, scale, random_bits);
            const uint16_t raw = *reinterpret_cast<const uint16_t*>(&packed4);
            packed_values |=
                static_cast<uint64_t>(raw) << ((element / 4) * 16);
            random_bits = fused_sr_next(random_bits);
        }
    } else {
        #pragma unroll
        for (int pair = 0; pair < NVFP4_BLOCK / 2; ++pair) {
            const float2 scaled = {
                values[pair * 2] * coefficient,
                values[pair * 2 + 1] * coefficient,
            };
            const uint8_t packed_pair = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(
                    scaled, __NV_E2M1, cudaRoundNearest));
            packed_values |= static_cast<uint64_t>(packed_pair) << (pair * 8);
        }
    }
    *reinterpret_cast<uint64_t*>(
        reinterpret_cast<uint8_t*>(row_fp4) + packed_offset) = packed_values;
    return block_sum;
}

__device__ __forceinline__ uint8_t mxfp4_e8m0_guarded_floor(
    float value,
    float floor_ratio) {
    if (value <= 1.0e-38f) return 0;
    const uint32_t bits = __float_as_uint(value);
    const uint8_t exponent = static_cast<uint8_t>((bits >> 23) & 0xffu);
    const uint32_t mantissa = bits & 0x7fffffu;
    if (mantissa == 0 || exponent >= 0xfeu) return exponent;
    if (floor_ratio > 1.0f && exponent > 0) {
        const float floor_value = __uint_as_float(
            static_cast<uint32_t>(exponent) << 23);
        if (value <= floor_value * floor_ratio) return exponent;
    }
    return static_cast<uint8_t>(exponent + 1);
}

__device__ __forceinline__ uint8_t mxfp4_e8m0_ceil(float value) {
    if (value <= 1.0e-38f) return 0;
    const uint32_t bits = __float_as_uint(value);
    uint8_t exponent = static_cast<uint8_t>((bits >> 23) & 0xffu);
    const uint32_t mantissa = bits & 0x7fffffu;
    if (mantissa != 0 && exponent < 0xfeu) ++exponent;
    return exponent;
}

__device__ __forceinline__ uint8_t mxfp4_e8m0_guarded_floor_log2(
    float log2_value,
    float floor_log2_ratio) {
    if (!isfinite(log2_value) || log2_value <= -126.0f) return 0;
    const float floor_exponent = floorf(log2_value);
    const float fraction = log2_value - floor_exponent;
    float selected_exponent = floor_exponent;
    if (fraction != 0.0f &&
        (floor_log2_ratio <= 0.0f || fraction > floor_log2_ratio)) {
        selected_exponent += 1.0f;
    }
    int code = static_cast<int>(selected_exponent) + 127;
    code = code < 1 ? 1 : (code > 254 ? 254 : code);
    return static_cast<uint8_t>(code);
}

__device__ __forceinline__ uint8_t positive_e2m1_code_from_log2(
    float value) {
    if (!isfinite(value) || value < -2.0f) return 0;
    if (value < -0.4150374992788438f) return 1;
    if (value < 0.32192809488736235f) return 2;
    if (value < 0.8073549220576041f) return 3;
    if (value < 1.3219280948873624f) return 4;
    if (value < 1.8073549220576042f) return 5;
    if (value < 2.321928094887362f) return 6;
    return 7;
}

__device__ __forceinline__ float mxfp4_quant_coeff(uint8_t scale) {
    if (scale == 0) return 0.0f;
    if (scale == 1) return __uint_as_float(0x7f800000u);
    return __uint_as_float(
        (static_cast<uint32_t>(256 - scale) << 23) | 0x00400000u);
}

__device__ __forceinline__ int mxfp4_scale_offset(
    int row,
    int block,
    int tiles_per_row) {
    const int tile_row = row / NVFP4_TILE;
    const int row_in_tile = row & (NVFP4_TILE - 1);
    const int tile_col = block / (NVFP4_TILE / MXFP4_BLOCK);
    const int scale_index = block & ((NVFP4_TILE / MXFP4_BLOCK) - 1);
    const int lane = row_in_tile & 31;
    const int group = row_in_tile >> 5;
    return (tile_row * tiles_per_row + tile_col) * 512 +
        lane * 16 + group * 4 + scale_index;
}

template <int TOPK>
__device__ __forceinline__ void insert_topk(
    float value,
    int index,
    float* values,
    int* indices);

template <
    bool DATA_SR,
    typename LOGIT_T,
    bool CENTERED_LOGITS = false,
    int TAIL_MATH = MXFP4_TAIL_NATIVE,
    int TOP_CANDIDATES = 0>
__device__ __forceinline__ float quantize_mxfp4_exp_block(
    LOGIT_T* __restrict__ logits,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    int row,
    int block,
    int N,
    int vocab_size,
    float row_max,
    float logit_temperature,
    float scale_floor_ratio,
    float scale_floor_log2_ratio,
    uint32_t sr_nonce,
    const __nv_bfloat16* __restrict__ logit_centers = nullptr,
    int center_stride = 0,
    int target = -1,
    float* local_top_values = nullptr,
    int* local_top_indices = nullptr,
    float* exact_block_max_out = nullptr,
    float* exact_block_sumexp_out = nullptr) {
    const int col0 = block * MXFP4_BLOCK;
    const int64_t row_offset = static_cast<int64_t>(row) * N;
    __nv_bfloat162 values[MXFP4_BLOCK / 2];
    float block_amax = 0.0f;
    float block_sum = 0.0f;
    float block_max_delta = -INFINITY;
    float exact_block_max = -INFINITY;
    float exact_block_sumexp = 0.0f;
    if constexpr (std::is_same_v<LOGIT_T, __nv_bfloat16>) {
        const auto* packed_logits = reinterpret_cast<const NVFP4BFloat16x8*>(
            logits + row_offset + col0);
        #pragma unroll
        for (int vector = 0; vector < MXFP4_BLOCK / 8; ++vector) {
            const NVFP4BFloat16x8 packed = packed_logits[vector];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int element = vector * 8 + pair * 2;
                const int col = col0 + element;
                const float2 raw = __bfloat1622float2(packed.values[pair]);
                float delta0 = -INFINITY;
                float delta1 = -INFINITY;
                if (col < vocab_size) {
                    const float z0 = logit_temperature * raw.x;
                    delta0 = z0 - row_max;
                    if constexpr (TOP_CANDIDATES > 0) {
                        if (col != target) {
                            insert_topk<TOP_CANDIDATES>(
                                z0,
                                col,
                                local_top_values,
                                local_top_indices);
                        }
                    }
                }
                if (col + 1 < vocab_size) {
                    const float z1 = logit_temperature * raw.y;
                    delta1 = z1 - row_max;
                    if constexpr (TOP_CANDIDATES > 0) {
                        if (col + 1 != target) {
                            insert_topk<TOP_CANDIDATES>(
                                z1,
                                col + 1,
                                local_top_values,
                                local_top_indices);
                        }
                    }
                }
                if constexpr (
                    TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED ||
                    TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                    if constexpr (
                        TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                        values[element / 2] = packed.values[pair];
                    } else {
                        values[element / 2] =
                            __floats2bfloat162_rn(delta0, delta1);
                    }
                    block_max_delta = fmaxf(
                        block_max_delta, fmaxf(delta0, delta1));
                } else {
                    const float numerator0 = mxfp4_tail_exp<TAIL_MATH>(
                        delta0, element);
                    const float numerator1 = mxfp4_tail_exp<TAIL_MATH>(
                        delta1, element + 1);
                    block_sum += numerator0;
                    block_sum += numerator1;
                    values[element / 2] =
                        __floats2bfloat162_rn(numerator0, numerator1);
                    const float2 rounded =
                        __bfloat1622float2(values[element / 2]);
                    block_amax = fmaxf(
                        block_amax, fmaxf(rounded.x, rounded.y));
                }
            }
        }
    } else {
        static_assert(std::is_same_v<LOGIT_T, __nv_fp8_e4m3>);
        float center = 0.0f;
        if constexpr (CENTERED_LOGITS) {
            center = __bfloat162float(logit_centers[
                static_cast<int64_t>(row) * center_stride + col0 / 32]);
        }
        const auto* packed_logits = reinterpret_cast<const FP8x16*>(
            logits + row_offset + col0);
        #pragma unroll
        for (int vector = 0; vector < MXFP4_BLOCK / 16; ++vector) {
            const FP8x16 packed = packed_logits[vector];
            #pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const int element = vector * 16 + pair * 2;
                const int col = col0 + element;
                float delta0 = -INFINITY;
                float delta1 = -INFINITY;
                if (col < vocab_size) {
                    const float z0 = logit_temperature *
                        (fp8_from_byte(packed.values[pair * 2]) + center);
                    delta0 = z0 - row_max;
                    if constexpr (TOP_CANDIDATES > 0) {
                        if (col != target) {
                            insert_topk<TOP_CANDIDATES>(
                                z0,
                                col,
                                local_top_values,
                                local_top_indices);
                        }
                    }
                }
                if (col + 1 < vocab_size) {
                    const float z1 = logit_temperature *
                        (fp8_from_byte(packed.values[pair * 2 + 1]) + center);
                    delta1 = z1 - row_max;
                    if constexpr (TOP_CANDIDATES > 0) {
                        if (col + 1 != target) {
                            insert_topk<TOP_CANDIDATES>(
                                z1,
                                col + 1,
                                local_top_values,
                                local_top_indices);
                        }
                    }
                }
                if constexpr (
                    TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED ||
                    TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                    if constexpr (
                        TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                        values[element / 2] = __floats2bfloat162_rn(
                            fp8_from_byte(packed.values[pair * 2]) + center,
                            fp8_from_byte(packed.values[pair * 2 + 1]) + center);
                    } else {
                        values[element / 2] =
                            __floats2bfloat162_rn(delta0, delta1);
                    }
                    block_max_delta = fmaxf(
                        block_max_delta, fmaxf(delta0, delta1));
                } else {
                    const float numerator0 = mxfp4_tail_exp<TAIL_MATH>(
                        delta0, element);
                    const float numerator1 = mxfp4_tail_exp<TAIL_MATH>(
                        delta1, element + 1);
                    block_sum += numerator0;
                    block_sum += numerator1;
                    values[element / 2] =
                        __floats2bfloat162_rn(numerator0, numerator1);
                    const float2 rounded =
                        __bfloat1622float2(values[element / 2]);
                    block_amax = fmaxf(
                        block_amax, fmaxf(rounded.x, rounded.y));
                }
            }
        }
    }

    if constexpr (TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED) {
        block_amax = isfinite(block_max_delta)
            ? __expf(block_max_delta)
            : 0.0f;
    }
    if (exact_block_max_out != nullptr) {
        exact_block_max = -INFINITY;
        #pragma unroll
        for (int pair = 0; pair < MXFP4_BLOCK / 2; ++pair) {
            float2 delta = __bfloat1622float2(values[pair]);
            if constexpr (
                TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                delta = {
                    fmaf(logit_temperature, delta.x, -row_max),
                    fmaf(logit_temperature, delta.y, -row_max),
                };
            }
            exact_block_max = fmaxf(
                exact_block_max, fmaxf(delta.x, delta.y));
        }
        if (isfinite(exact_block_max)) {
            #pragma unroll
            for (int pair = 0; pair < MXFP4_BLOCK / 2; ++pair) {
                float2 delta = __bfloat1622float2(values[pair]);
                if constexpr (
                    TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                    delta = {
                        fmaf(logit_temperature, delta.x, -row_max),
                        fmaf(logit_temperature, delta.y, -row_max),
                    };
                }
                exact_block_sumexp +=
                    __expf(delta.x - exact_block_max) +
                    __expf(delta.y - exact_block_max);
            }
        }
    }

    constexpr float LOG2E = 1.4426950408889634f;
    const uint8_t scale =
        TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED
        ? mxfp4_e8m0_guarded_floor_log2(
              block_max_delta * LOG2E, scale_floor_log2_ratio)
        : (DATA_SR
               ? mxfp4_e8m0_ceil(block_amax)
               : mxfp4_e8m0_guarded_floor(
                     block_amax, scale_floor_ratio));
    row_sc[mxfp4_scale_offset(row, block, N / NVFP4_TILE)] = scale;
    const float coefficient = mxfp4_quant_coeff(scale);
    const int64_t packed_offset =
        static_cast<int64_t>(row) * (N / 2) + col0 / 2;
    uint4 packed_values{};
    auto* packed_bytes = reinterpret_cast<uint8_t*>(&packed_values);
    constexpr float LOG2_FP4_MAX = 2.584962500721156f;
    constexpr float AFFINE_A = 1.62330034f;
    constexpr float AFFINE_B = 0.92083546f;
    const float log2_coefficient = coefficient > 0.0f
        ? LOG2_FP4_MAX + 127.0f - static_cast<float>(scale)
        : -INFINITY;
    if constexpr (TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
        #pragma unroll
        for (int pair = 0; pair < MXFP4_BLOCK / 2; ++pair) {
            const float2 raw = __bfloat1622float2(values[pair]);
            const float2 delta = {
                fmaf(logit_temperature, raw.x, -row_max),
                fmaf(logit_temperature, raw.y, -row_max),
            };
            const uint8_t code0 = positive_e2m1_code_from_log2(
                fmaf(LOG2E, delta.x, log2_coefficient));
            const uint8_t code1 = positive_e2m1_code_from_log2(
                fmaf(LOG2E, delta.y, log2_coefficient));
            packed_bytes[pair] = static_cast<uint8_t>(code0 | (code1 << 4));
        }
    } else if constexpr (DATA_SR) {
        const uint32_t block_key =
            static_cast<uint32_t>(row) * (N / MXFP4_BLOCK) + block;
        uint32_t random_bits = fused_sr_bits(sr_nonce + block_key);
        #pragma unroll
        for (int element = 0; element < MXFP4_BLOCK; element += 4) {
            const float2 in01 =
                __bfloat1622float2(values[element / 2]);
            const float2 in23 =
                __bfloat1622float2(values[element / 2 + 1]);
            float2 converted01 = in01;
            float2 converted23 = in23;
            float conversion_scale = coefficient;
            if constexpr (TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED) {
                converted01 = {
                    fmaxf(fmaf(
                        AFFINE_A,
                        fmaf(LOG2E, in01.x, log2_coefficient),
                        AFFINE_B), 0.0f),
                    fmaxf(fmaf(
                        AFFINE_A,
                        fmaf(LOG2E, in01.y, log2_coefficient),
                        AFFINE_B), 0.0f),
                };
                converted23 = {
                    fmaxf(fmaf(
                        AFFINE_A,
                        fmaf(LOG2E, in23.x, log2_coefficient),
                        AFFINE_B), 0.0f),
                    fmaxf(fmaf(
                        AFFINE_A,
                        fmaf(LOG2E, in23.y, log2_coefficient),
                        AFFINE_B), 0.0f),
                };
                conversion_scale = coefficient > 0.0f ? 1.0f : 0.0f;
            }
            const float2 scale_pair = {conversion_scale, conversion_scale};
            const auto packed4 =
                transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                    converted01, converted23, scale_pair, random_bits);
            const uint16_t raw =
                *reinterpret_cast<const uint16_t*>(&packed4);
            packed_bytes[element / 2] = static_cast<uint8_t>(raw & 0xffu);
            packed_bytes[element / 2 + 1] =
                static_cast<uint8_t>((raw >> 8) & 0xffu);
            random_bits = fused_sr_next(random_bits);
        }
    } else {
        #pragma unroll
        for (int pair = 0; pair < MXFP4_BLOCK / 2; ++pair) {
            float2 scaled = __bfloat1622float2(values[pair]);
            if constexpr (TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED) {
                scaled = coefficient > 0.0f
                    ? float2{
                          fmaxf(fmaf(
                              AFFINE_A,
                              fmaf(LOG2E, scaled.x, log2_coefficient),
                              AFFINE_B), 0.0f),
                          fmaxf(fmaf(
                              AFFINE_A,
                              fmaf(LOG2E, scaled.y, log2_coefficient),
                              AFFINE_B), 0.0f),
                      }
                    : float2{0.0f, 0.0f};
            } else {
                scaled.x *= coefficient;
                scaled.y *= coefficient;
            }
            packed_bytes[pair] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(
                    scaled, __NV_E2M1, cudaRoundNearest));
        }
    }
    *reinterpret_cast<uint4*>(
        reinterpret_cast<uint8_t*>(row_fp4) + packed_offset) = packed_values;
    if (exact_block_max_out != nullptr) {
        *exact_block_max_out = exact_block_max;
    }
    if (exact_block_sumexp_out != nullptr) {
        *exact_block_sumexp_out = exact_block_sumexp;
    }
    if constexpr (
        TAIL_MATH == MXFP4_TAIL_POLY4_REPRESENTED ||
        TAIL_MATH == MXFP4_TAIL_AFFINE_REPRESENTED ||
        TAIL_MATH == MXFP4_TAIL_HYBRID25_REPRESENTED ||
        TAIL_MATH == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
        int represented_sum_x2 = 0;
        const auto* packed_words = reinterpret_cast<const uint32_t*>(
            &packed_values);
        #pragma unroll
        for (int word = 0; word < MXFP4_BLOCK / 8; ++word) {
            represented_sum_x2 = mxfp4_qsum_x2_from_word(
                packed_words[word], represented_sum_x2);
        }
        return coefficient > 0.0f
            ? (0.5f * static_cast<float>(represented_sum_x2)) / coefficient
            : 0.0f;
    }
    return block_sum;
}

template <bool DATA_SR>
__device__ __forceinline__ float quantize_mxfp8_exp_block(
    __nv_bfloat16* __restrict__ logits,
    __nv_fp4x2_e2m1* __restrict__ row_payload,
    uint8_t* __restrict__ row_sc,
    int row,
    int block,
    int N,
    int vocab_size,
    float row_max,
    float logit_temperature,
    float quant_max,
    uint32_t sr_nonce) {
    const int col0 = block * MXFP4_BLOCK;
    const int64_t row_offset = static_cast<int64_t>(row) * N;
    float values[MXFP4_BLOCK];
    float block_amax = 0.0f;
    float block_sum = 0.0f;
    const auto* packed_logits = reinterpret_cast<const NVFP4BFloat16x8*>(
        logits + row_offset + col0);
    #pragma unroll
    for (int vector = 0; vector < MXFP4_BLOCK / 8; ++vector) {
        const NVFP4BFloat16x8 packed = packed_logits[vector];
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int element = vector * 8 + pair * 2;
            const int col = col0 + element;
            const float2 raw = __bfloat1622float2(packed.values[pair]);
            float numerator0 = 0.0f;
            float numerator1 = 0.0f;
            if (col < vocab_size) {
                numerator0 = __expf(fmaf(logit_temperature, raw.x, -row_max));
                block_sum += numerator0;
            }
            if (col + 1 < vocab_size) {
                numerator1 = __expf(fmaf(logit_temperature, raw.y, -row_max));
                block_sum += numerator1;
            }
            const float rounded0 = __bfloat162float(__float2bfloat16(numerator0));
            const float rounded1 = __bfloat162float(__float2bfloat16(numerator1));
            values[element] = rounded0;
            values[element + 1] = rounded1;
            block_amax = fmaxf(block_amax, fmaxf(rounded0, rounded1));
        }
    }

    __nv_fp8_e8m0 scale;
    scale.__x = __nv_cvt_float_to_e8m0(
        fmaxf(block_amax / quant_max, 1.0e-12f),
        __NV_SATFINITE,
        cudaRoundPosInf);
    row_sc[mxfp4_scale_offset(row, block, N / NVFP4_TILE)] = scale.__x;
    const float scale_inv = 1.0f / static_cast<float>(scale);

    uint4 packed_output[2]{};
    auto* output_bytes = reinterpret_cast<uint8_t*>(packed_output);
    if constexpr (DATA_SR) {
        const uint32_t block_key =
            static_cast<uint32_t>(row) * (N / MXFP4_BLOCK) + block;
        uint32_t random_bits = fused_sr_bits(sr_nonce + block_key);
        #pragma unroll
        for (int element = 0; element < MXFP4_BLOCK; element += 4) {
            const float value0 = values[element] * scale_inv;
            const float value1 = values[element + 1] * scale_inv;
            const float value2 = values[element + 2] * scale_inv;
            const float value3 = values[element + 3] * scale_inv;
            uint32_t packed;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            asm volatile(
                "cvt.rs.satfinite.e4m3x4.f32 %0, {%4, %3, %2, %1}, %5;"
                : "=r"(packed)
                : "f"(value0), "f"(value1), "f"(value2), "f"(value3),
                  "r"(random_bits));
#else
            auto* bytes = reinterpret_cast<uint8_t*>(&packed);
            bytes[0] = static_cast<__nv_fp8_e4m3>(value0).__x;
            bytes[1] = static_cast<__nv_fp8_e4m3>(value1).__x;
            bytes[2] = static_cast<__nv_fp8_e4m3>(value2).__x;
            bytes[3] = static_cast<__nv_fp8_e4m3>(value3).__x;
#endif
            reinterpret_cast<uint32_t*>(output_bytes)[element / 4] = packed;
            random_bits = fused_sr_next(random_bits);
        }
    } else {
        #pragma unroll
        for (int element = 0; element < MXFP4_BLOCK; ++element) {
            const __nv_fp8_e4m3 quantized(values[element] * scale_inv);
            output_bytes[element] = quantized.__x;
        }
    }
    auto* output_ptr = reinterpret_cast<uint8_t*>(row_payload) +
        row_offset + col0;
    reinterpret_cast<uint4*>(output_ptr)[0] = packed_output[0];
    reinterpret_cast<uint4*>(output_ptr)[1] = packed_output[1];
    return block_sum;
}

__global__ void make_nvfp4_row_normalization_kernel(
    const float* __restrict__ row_max,
    const float* __restrict__ lse,
    float* __restrict__ row_normalization,
    int M) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < M;
         row += blockDim.x * gridDim.x) {
        row_normalization[row] = __expf(row_max[row] - lse[row]);
    }
}

#define CUBLAS_CHECK(status)                                                   \
    do {                                                                       \
        const cublasStatus_t status_ = (status);                               \
        TORCH_CHECK(                                                           \
            status_ == CUBLAS_STATUS_SUCCESS,                                  \
            "cuBLAS error: ",                                                  \
            cublasGetStatusString(status_));                                   \
    } while (0)

class CublasPointerModeGuard {
public:
    CublasPointerModeGuard(cublasHandle_t handle, cublasPointerMode_t mode)
        : handle_(handle) {
        CUBLAS_CHECK(cublasGetPointerMode(handle_, &previous_));
        CUBLAS_CHECK(cublasSetPointerMode(handle_, mode));
    }

    ~CublasPointerModeGuard() {
        cublasSetPointerMode(handle_, previous_);
    }

private:
    cublasHandle_t handle_;
    cublasPointerMode_t previous_;
};

struct alignas(16) BFloat16x8 {
    __nv_bfloat162 values[4];
};

__device__ __forceinline__ void unpack_bfloat16x8(
    const BFloat16x8& packed,
    float (&values)[8]
) {
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 converted = __bfloat1622float2(packed.values[pair]);
        values[pair * 2] = converted.x;
        values[pair * 2 + 1] = converted.y;
    }
}

__global__ void make_backward_coefficients_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ valid_count,
    float* __restrict__ coefficients) {
    coefficients[0] = grad_output[0] / fmaxf(valid_count[0], 1.0f);
    coefficients[1] = 0.0f;
}

__global__ void valid_mask_count_kernel(
    const int64_t* __restrict__ targets,
    bool* __restrict__ valid,
    float* __restrict__ valid_count,
    int64_t numel,
    int64_t ignore_index) {
    int local_count = 0;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         index < numel;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const bool is_valid = targets[index] != ignore_index;
        valid[index] = is_valid;
        local_count += static_cast<int>(is_valid);
    }
    local_count = __reduce_add_sync(0xffffffff, local_count);
    if ((threadIdx.x & 31) == 0 && local_count != 0) {
        atomicAdd(valid_count, static_cast<float>(local_count));
    }
}

__device__ __forceinline__ float warp_reduce_sum(float v);

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void replace_target_logits_bf16_kernel(
    __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const int64_t* __restrict__ targets,
    int M,
    int N,
    int K,
    int vocab_size,
    int64_t ignore_index) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    const int64_t target64 = targets[row];
    if (target64 == ignore_index || target64 < 0 || target64 >= vocab_size) {
        return;
    }
    const int target = static_cast<int>(target64);
    float local_sum = 0.0f;
    for (int col = tid; col < K; col += THREADS) {
        local_sum +=
            __bfloat162float(x[static_cast<int64_t>(row) * K + col]) *
            __bfloat162float(weight[static_cast<int64_t>(target) * K + col]);
    }
    local_sum = warp_reduce_sum(local_sum);

    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < WARPS ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            logits[static_cast<int64_t>(row) * N + target] =
                __float2bfloat16(sum);
        }
    }
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

__device__ __forceinline__ void warp_reduce_argmax(float& value, int& index) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value =
            __shfl_down_sync(0xffffffff, value, offset);
        const int other_index =
            __shfl_down_sync(0xffffffff, index, offset);
        if (other_value > value ||
            (other_value == value && other_index >= 0 &&
             (index < 0 || other_index < index))) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ bool argmax_better(
    float candidate_value,
    int candidate_index,
    float value,
    int index) {
    return candidate_value > value ||
        (candidate_value == value && candidate_index >= 0 &&
         (index < 0 || candidate_index < index));
}

__device__ __forceinline__ void insert_top4(
    float value,
    int index,
    float* values,
    int* indices) {
    #pragma unroll
    for (int rank = 0; rank < MAX_TOPK_SPLIT; ++rank) {
        if (argmax_better(value, index, values[rank], indices[rank])) {
            #pragma unroll
            for (int shift = MAX_TOPK_SPLIT - 1; shift > rank; --shift) {
                values[shift] = values[shift - 1];
                indices[shift] = indices[shift - 1];
            }
            values[rank] = value;
            indices[rank] = index;
            return;
        }
    }
}

template <int TOPK>
__device__ __forceinline__ void insert_topk(
    float value,
    int index,
    float* values,
    int* indices) {
    #pragma unroll
    for (int rank = 0; rank < TOPK; ++rank) {
        if (argmax_better(value, index, values[rank], indices[rank])) {
            #pragma unroll
            for (int shift = TOPK - 1; shift > rank; --shift) {
                values[shift] = values[shift - 1];
                indices[shift] = indices[shift - 1];
            }
            values[rank] = value;
            indices[rank] = index;
            return;
        }
    }
}

inline int select_row_threads(const char* env_value, int64_t vocab_size) {
    if (env_value != nullptr) {
        if (env_value[0] == '1') return 1024;
        if (env_value[0] == '5') return 512;
        if (env_value[0] == '2') return 256;
    }
    if (vocab_size >= 65536) return 1024;
    if (vocab_size >= 32768) return 512;
    return 256;
}

inline bool use_stage_exp(const char* env_value) {
    return env_value != nullptr && env_value[0] != '0';
}

template <int THREADS, bool STAGE_EXP>
__global__ __launch_bounds__(THREADS)
void softmax_loss_probs_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    __nv_bfloat16* __restrict__ probs,
    float* __restrict__ target_probs,
    float* __restrict__ topk_probs,
    int* __restrict__ topk_indices,
    float* __restrict__ loss_sum,
    float* __restrict__ count_sum,
    int M,
    int N,
    int vocab_size,
    bool grad_cache,
    bool target_split,
    int topk_split) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_red[ROW_WARPS];
    __shared__ int warp_idx[ROW_WARPS];
    __shared__ float s_max;
    __shared__ float s_inv_sum;
    __shared__ float s_lse;
    __shared__ int s_topk_indices[MAX_TOPK_SPLIT];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bool is_valid = valid[row];
    const int target = is_valid ? static_cast<int>(targets[row]) : -1;

    float local_max = -INFINITY;
    float local_top_values[MAX_TOPK_SPLIT] = {
        -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int local_top_indices[MAX_TOPK_SPLIT] = {-1, -1, -1, -1};
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z =
                __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
            local_max = fmaxf(local_max, z);
            if (topk_split > 0 && col != target) {
                insert_top4(
                    z,
                    col,
                    local_top_values,
                    local_top_indices);
            }
        }
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    if (topk_split > 0) {
        #pragma unroll
        for (int rank = 0; rank < MAX_TOPK_SPLIT; ++rank) {
            if (rank < topk_split) {
                float candidate_value = -INFINITY;
                int candidate_index = -1;
                #pragma unroll
                for (int local_rank = 0;
                     local_rank < MAX_TOPK_SPLIT;
                     ++local_rank) {
                    const int index = local_top_indices[local_rank];
                    bool selected = false;
                    #pragma unroll
                    for (int prior = 0; prior < MAX_TOPK_SPLIT; ++prior) {
                        if (prior < rank && index == s_topk_indices[prior]) {
                            selected = true;
                        }
                    }
                    if (!selected &&
                        argmax_better(
                            local_top_values[local_rank],
                            index,
                            candidate_value,
                            candidate_index)) {
                        candidate_value = local_top_values[local_rank];
                        candidate_index = index;
                    }
                }
                warp_reduce_argmax(candidate_value, candidate_index);
                if (lane == 0) {
                    warp_red[warp] = candidate_value;
                    warp_idx[warp] = candidate_index;
                }
                __syncthreads();
                if (warp == 0) {
                    float block_value =
                        (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
                    int block_index =
                        (lane < ROW_WARPS) ? warp_idx[lane] : -1;
                    warp_reduce_argmax(block_value, block_index);
                    if (lane == 0) {
                        s_topk_indices[rank] = block_index;
                    }
                }
                __syncthreads();
            }
        }
    }

    float local_sum = 0.0f;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
            const float e = __expf(z - s_max);
            local_sum += e;
            if constexpr (STAGE_EXP) {
                probs[static_cast<int64_t>(row) * N + col] = __float2bfloat16(e);
            }
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS) ? warp_red[lane] : 0.0f;
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

    for (int col = tid; col < N; col += THREADS) {
        float p = 0.0f;
        if (is_valid && col < vocab_size) {
            if constexpr (STAGE_EXP) {
                p = __bfloat162float(probs[static_cast<int64_t>(row) * N + col]) * s_inv_sum;
            } else {
                const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
                p = __expf(z - s_max) * s_inv_sum;
            }
            if (grad_cache && col == target) {
                p -= 1.0f;
            }
        }
        __nv_bfloat16 p_out = __float2bfloat16(p);
        int split_rank = -1;
        #pragma unroll
        for (int rank = 0; rank < MAX_TOPK_SPLIT; ++rank) {
            if (rank < topk_split && col == s_topk_indices[rank]) {
                split_rank = rank;
            }
        }
        if (target_split && is_valid && col == target) {
            target_probs[row] = __bfloat162float(p_out);
            p_out = __float2bfloat16(0.0f);
        } else if (topk_split > 0 && is_valid && split_rank >= 0) {
            const int64_t split_offset =
                static_cast<int64_t>(row) * topk_split + split_rank;
            topk_probs[split_offset] = __bfloat162float(p_out);
            topk_indices[split_offset] = col;
            p_out = __float2bfloat16(0.0f);
        }
        probs[static_cast<int64_t>(row) * N + col] = p_out;
    }
    if (!is_valid && tid == 0) {
        if (target_split) {
            target_probs[row] = 0.0f;
        }
        for (int rank = 0; rank < topk_split; ++rank) {
            const int64_t split_offset =
                static_cast<int64_t>(row) * topk_split + rank;
            topk_probs[split_offset] = 0.0f;
            topk_indices[split_offset] = -1;
        }
    } else if (topk_split > 0 && tid == 0) {
        for (int rank = 0; rank < topk_split; ++rank) {
            if (s_topk_indices[rank] < 0) {
                const int64_t split_offset =
                    static_cast<int64_t>(row) * topk_split + rank;
                topk_probs[split_offset] = 0.0f;
                topk_indices[split_offset] = -1;
            }
        }
    }
}

template <int THREADS, bool STAGE_EXP>
__global__ __launch_bounds__(THREADS)
void softmax_loss_grad_probs_all_valid_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    __nv_bfloat16* __restrict__ grad_probs,
    float* __restrict__ loss_sum,
    int M,
    int N) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_red[ROW_WARPS];
    __shared__ float s_max;
    __shared__ float s_inv_sum;
    __shared__ float s_lse;
    __shared__ int64_t s_target;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t row_offset = static_cast<int64_t>(row) * N;

    float local_max = -INFINITY;
    for (int col = tid; col < N; col += THREADS) {
        local_max = fmaxf(local_max, __bfloat162float(logits[row_offset + col]));
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col < N; col += THREADS) {
        const float z = __bfloat162float(logits[row_offset + col]);
        const float e = __expf(z - s_max);
        local_sum += e;
        if constexpr (STAGE_EXP) {
            grad_probs[row_offset + col] = __float2bfloat16(e);
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS) ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            s_inv_sum = 1.0f / sum;
            s_lse = s_max + logf(sum);
            const int64_t target = targets[row];
            s_target = target;
            const float target_logit = __bfloat162float(logits[row_offset + target]);
            atomicAdd(loss_sum, s_lse - target_logit);
        }
    }
    __syncthreads();

    for (int col = tid; col < N; col += THREADS) {
        float p;
        if constexpr (STAGE_EXP) {
            p = __bfloat162float(grad_probs[row_offset + col]) * s_inv_sum;
        } else {
            const float z = __bfloat162float(logits[row_offset + col]);
            p = __expf(z - s_max) * s_inv_sum;
        }
        if (col == s_target) {
            p -= 1.0f;
        }
        grad_probs[row_offset + col] = __float2bfloat16(p);
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_loss_lse_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    float* __restrict__ loss_sum,
    float* __restrict__ count_sum,
    int M,
    int N,
    int vocab_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_red[ROW_WARPS];
    __shared__ float s_max;
    __shared__ float s_lse;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bool is_valid = valid[row];

    float local_max = -INFINITY;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            local_max = fmaxf(local_max, __bfloat162float(logits[static_cast<int64_t>(row) * N + col]));
        }
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
            local_sum += __expf(z - s_max);
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS) ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            s_lse = (sum > 0.0f) ? (s_max + logf(sum)) : 0.0f;
            lse[row] = s_lse;
            if (is_valid) {
                const int64_t target = targets[row];
                const float target_logit = __bfloat162float(logits[static_cast<int64_t>(row) * N + target]);
                atomicAdd(loss_sum, s_lse - target_logit);
                atomicAdd(count_sum, 1.0f);
            }
        }
    }
}

template <int THREADS, int TOPK_SPLIT, int TOPK_OUTPUT_STRIDE>
__global__ __launch_bounds__(THREADS)
void softmax_loss_lse_target_topk_split_kernel(
    __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    float* __restrict__ target_probs,
    float* __restrict__ topk_probs,
    int* __restrict__ topk_indices,
    float* __restrict__ loss_sum,
    float* __restrict__ count_sum,
    int M,
    int N,
    int vocab_size,
    float logit_temperature) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_red[ROW_WARPS];
    __shared__ int warp_idx[ROW_WARPS];
    __shared__ float s_max;
    __shared__ float s_lse;
    __shared__ int s_topk_indices[TOPK_SPLIT > 0 ? TOPK_SPLIT : 1];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bool is_valid = valid[row];
    const int target = is_valid ? static_cast<int>(targets[row]) : -1;
    const int64_t row_offset = static_cast<int64_t>(row) * N;

    float local_max = -INFINITY;
    float local_top_values[TOPK_SPLIT > 0 ? TOPK_SPLIT : 1];
    int local_top_indices[TOPK_SPLIT > 0 ? TOPK_SPLIT : 1];
    #pragma unroll
    for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
        local_top_values[rank] = -INFINITY;
        local_top_indices[rank] = -1;
    }
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z = logit_temperature *
                __bfloat162float(logits[row_offset + col]);
            local_max = fmaxf(local_max, z);
            if constexpr (TOPK_SPLIT > 0) {
                if (col != target) {
                    insert_topk<TOPK_SPLIT>(
                        z, col, local_top_values, local_top_indices);
                }
            }
        }
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    #pragma unroll
    for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
        float candidate_value = -INFINITY;
        int candidate_index = -1;
        #pragma unroll
        for (int local_rank = 0; local_rank < TOPK_SPLIT; ++local_rank) {
            const int index = local_top_indices[local_rank];
            bool selected = false;
            #pragma unroll
            for (int prior = 0; prior < rank; ++prior) {
                if (index == s_topk_indices[prior]) {
                    selected = true;
                }
            }
            if (!selected &&
                argmax_better(
                    local_top_values[local_rank],
                    index,
                    candidate_value,
                    candidate_index)) {
                candidate_value = local_top_values[local_rank];
                candidate_index = index;
            }
        }
        warp_reduce_argmax(candidate_value, candidate_index);
        if (lane == 0) {
            warp_red[warp] = candidate_value;
            warp_idx[warp] = candidate_index;
        }
        __syncthreads();
        if (warp == 0) {
            float block_value =
                (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
            int block_index =
                (lane < ROW_WARPS) ? warp_idx[lane] : -1;
            warp_reduce_argmax(block_value, block_index);
            if (lane == 0) s_topk_indices[rank] = block_index;
        }
        __syncthreads();
    }

    float local_sum = 0.0f;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z = logit_temperature *
                __bfloat162float(logits[row_offset + col]);
            local_sum += __expf(z - s_max);
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS) ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            s_lse = (sum > 0.0f) ? (s_max + logf(sum)) : 0.0f;
            lse[row] = s_lse;
            if (is_valid) {
                const float target_logit = logit_temperature *
                    __bfloat162float(logits[row_offset + target]);
                target_probs[row] = __expf(target_logit - s_lse);
                atomicAdd(loss_sum, s_lse - target_logit);
                atomicAdd(count_sum, 1.0f);
                logits[row_offset + target] = __float2bfloat16(-INFINITY);
                #pragma unroll
                for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
                    const int index = s_topk_indices[rank];
                    const int64_t output =
                        static_cast<int64_t>(row) * TOPK_OUTPUT_STRIDE + rank;
                    if (index >= 0) {
                        const float z = logit_temperature *
                            __bfloat162float(logits[row_offset + index]);
                        topk_probs[output] = __expf(z - s_lse);
                        topk_indices[output] = index;
                        logits[row_offset + index] =
                            __float2bfloat16(-INFINITY);
                    } else {
                        topk_probs[output] = 0.0f;
                        topk_indices[output] = 0;
                    }
                }
            } else {
                target_probs[row] = 0.0f;
                #pragma unroll
                for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
                    const int64_t output =
                        static_cast<int64_t>(row) * TOPK_OUTPUT_STRIDE + rank;
                    topk_probs[output] = 0.0f;
                    topk_indices[output] = 0;
                }
            }
        }
    }
}

template <
    int THREADS,
    int OUTPUT_TOPK,
    int FUSED_ROW_FORMAT = FUSED_ROW_NONE,
    bool FUSED_ROW_DATA_SR = false,
    typename LOGIT_T = __nv_bfloat16,
    bool CENTERED_LOGITS = false,
    int MXFP4_TAIL_MATH_MODE = MXFP4_TAIL_NATIVE,
    bool MXFP4_SINGLE_PASS = false,
    bool FAST_TOP12 = false,
    bool MID_TOP12 = false>
__global__ __launch_bounds__(THREADS)
void softmax_loss_lse_target_hierarchical_topk_kernel(
    LOGIT_T* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    float* __restrict__ target_probs,
    float* __restrict__ topk_probs,
    int* __restrict__ topk_indices,
    float* __restrict__ loss_sum,
    float* __restrict__ count_sum,
    int M,
    int N,
    int vocab_size,
    float logit_temperature,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    float* __restrict__ row_max,
    float* __restrict__ row_sg,
    float row_quant_parameter,
    float fixed_anchor,
    const uint64_t* __restrict__ row_rng_state = nullptr,
    const __nv_bfloat16* __restrict__ logit_centers = nullptr,
    int logit_center_stride = 0) {
    static_assert(
        OUTPUT_TOPK == 0 || OUTPUT_TOPK == 8 || OUTPUT_TOPK == 12 ||
            OUTPUT_TOPK == 16);
    static_assert(
        FUSED_ROW_FORMAT == FUSED_ROW_NONE ||
            FUSED_ROW_FORMAT == FUSED_ROW_NVFP4 ||
            FUSED_ROW_FORMAT == FUSED_ROW_MXFP4 ||
            FUSED_ROW_FORMAT == FUSED_ROW_MXFP8);
    static_assert(
        !CENTERED_LOGITS || std::is_same_v<LOGIT_T, __nv_fp8_e4m3>);
    static_assert(
        MXFP4_TAIL_MATH_MODE >= MXFP4_TAIL_NATIVE &&
        MXFP4_TAIL_MATH_MODE <= MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED);
    static_assert(
        !MXFP4_SINGLE_PASS ||
            (FUSED_ROW_FORMAT == FUSED_ROW_MXFP4 &&
             std::is_same_v<LOGIT_T, __nv_bfloat16> &&
             !CENTERED_LOGITS &&
             (MXFP4_TAIL_MATH_MODE == MXFP4_TAIL_NATIVE ||
              MXFP4_TAIL_MATH_MODE ==
                  MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED)),
        "single-pass MXFP4 production requires native or log-code BF16 logits");
    static_assert(
        !FAST_TOP12 ||
            (THREADS == 128 && OUTPUT_TOPK == 12 &&
             FUSED_ROW_FORMAT == FUSED_ROW_MXFP4 && !MXFP4_SINGLE_PASS),
        "fast top-12 selection requires the 128-thread MXFP4 producer");
    static_assert(
        !MID_TOP12 ||
            (THREADS == 128 && OUTPUT_TOPK == 12 &&
             FUSED_ROW_FORMAT == FUSED_ROW_MXFP4 && !MXFP4_SINGLE_PASS &&
             !FAST_TOP12),
        "mid top-12 selection requires the 128-thread MXFP4 producer");
    constexpr int ROW_WARPS = THREADS / 32;
    // Keep a small candidate hierarchy in registers/shared memory. This is a
    // high-recall selector; selected logits are recomputed exactly downstream.
    constexpr int LOCAL_CANDIDATES = 1;
    constexpr int WARP_CANDIDATES = OUTPUT_TOPK == 0
        ? 0
        : (FAST_TOP12
               ? 4
               : (MID_TOP12
               ? 6
               : (THREADS == 64
               ? (OUTPUT_TOPK + 1) / 2
               : (THREADS == 128
               ? 8
               : (OUTPUT_TOPK == 8 ? (THREADS == 256 ? 4 : 2) : 4)))));

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float warp_red[ROW_WARPS];
    __shared__ float s_max;
    __shared__ float s_lse;
    __shared__ float warp_top_values[ROW_WARPS][
        WARP_CANDIDATES > 0 ? WARP_CANDIDATES : 1];
    __shared__ int warp_top_indices[ROW_WARPS][
        WARP_CANDIDATES > 0 ? WARP_CANDIDATES : 1];
    __shared__ int s_topk_indices[OUTPUT_TOPK > 0 ? OUTPUT_TOPK : 1];
    __shared__ float s_selected_logits[OUTPUT_TOPK + 1];
    __shared__ uint32_t s_row_sr_nonce;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if constexpr (FUSED_ROW_DATA_SR) {
        if (tid == 0) {
            const uint64_t invocation = row_rng_state == nullptr
                ? 0
                : row_rng_state[0];
            s_row_sr_nonce = static_cast<uint32_t>(invocation) ^
                static_cast<uint32_t>(invocation >> 32);
        }
        __syncthreads();
    }
    const uint32_t row_sr_nonce = FUSED_ROW_DATA_SR
        ? s_row_sr_nonce
        : 0;
    const bool is_valid = valid[row];
    const int target = is_valid ? static_cast<int>(targets[row]) : -1;
    const int64_t row_offset = static_cast<int64_t>(row) * N;

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float local_top_values[LOCAL_CANDIDATES];
    int local_top_indices[LOCAL_CANDIDATES];
    #pragma unroll
    for (int rank = 0; rank < LOCAL_CANDIDATES; ++rank) {
        local_top_values[rank] = -INFINITY;
        local_top_indices[rank] = -1;
    }
    if constexpr (MXFP4_SINGLE_PASS) {
        const int blocks = N / MXFP4_BLOCK;
        const int active_vocab = is_valid ? vocab_size : 0;
        for (int block = tid; block < blocks; block += THREADS) {
            float block_max = -INFINITY;
            float block_sumexp = 0.0f;
            const float represented_sum = quantize_mxfp4_exp_block<
                FUSED_ROW_DATA_SR,
                LOGIT_T,
                false,
                MXFP4_TAIL_MATH_MODE,
                LOCAL_CANDIDATES>(
                    logits,
                    row_fp4,
                    row_sc,
                    row,
                    block,
                    N,
                    active_vocab,
                    fixed_anchor,
                    logit_temperature,
                    row_quant_parameter,
                    log2f(row_quant_parameter),
                    row_sr_nonce,
                    nullptr,
                    0,
                    target,
                    local_top_values,
                    local_top_indices,
                    MXFP4_TAIL_MATH_MODE ==
                            MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED
                        ? &block_max
                        : nullptr,
                    MXFP4_TAIL_MATH_MODE ==
                            MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED
                        ? &block_sumexp
                        : nullptr);
            if constexpr (
                MXFP4_TAIL_MATH_MODE ==
                    MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
                const float new_max = fmaxf(local_max, block_max);
                local_sum = local_sum * __expf(local_max - new_max) +
                    block_sumexp * __expf(block_max - new_max);
                local_max = new_max;
            } else {
                local_sum += represented_sum;
            }
        }
    } else if (is_valid) {
        if ((N & 1) == 0) {
            const int packed_vocab = vocab_size / 2;
            if constexpr (std::is_same_v<LOGIT_T, __nv_bfloat16>) {
                const auto* packed_logits =
                    reinterpret_cast<const __nv_bfloat162*>(
                        logits + row_offset);
                for (int pair = tid; pair < packed_vocab; pair += THREADS) {
                    const int col = pair * 2;
                    const float2 raw =
                        __bfloat1622float2(packed_logits[pair]);
                    const float z0 = logit_temperature * raw.x;
                    local_max = fmaxf(local_max, z0);
                    if constexpr (OUTPUT_TOPK > 0) {
                        if (col != target) {
                            insert_topk<LOCAL_CANDIDATES>(
                                z0, col, local_top_values, local_top_indices);
                        }
                    }
                    const float z1 = logit_temperature * raw.y;
                    local_max = fmaxf(local_max, z1);
                    if constexpr (OUTPUT_TOPK > 0) {
                        if (col + 1 != target) {
                            insert_topk<LOCAL_CANDIDATES>(
                                z1, col + 1,
                                local_top_values, local_top_indices);
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<LOGIT_T, __nv_fp8_e4m3>);
                const auto* packed_logits = reinterpret_cast<const uint16_t*>(
                    logits + row_offset);
                for (int pair = tid; pair < packed_vocab; pair += THREADS) {
                    const int col = pair * 2;
                    const uint16_t packed = packed_logits[pair];
                    float center = 0.0f;
                    if constexpr (CENTERED_LOGITS) {
                        center = __bfloat162float(logit_centers[
                            static_cast<int64_t>(row) *
                                    logit_center_stride +
                                col / 32]);
                    }
                    const float z0 = logit_temperature *
                        (fp8_from_byte(static_cast<uint8_t>(packed)) + center);
                    local_max = fmaxf(local_max, z0);
                    if constexpr (OUTPUT_TOPK > 0) {
                        if (col != target) {
                            insert_topk<LOCAL_CANDIDATES>(
                                z0, col, local_top_values, local_top_indices);
                        }
                    }
                    const float z1 = logit_temperature *
                        (fp8_from_byte(static_cast<uint8_t>(packed >> 8)) +
                         center);
                    local_max = fmaxf(local_max, z1);
                    if constexpr (OUTPUT_TOPK > 0) {
                        if (col + 1 != target) {
                            insert_topk<LOCAL_CANDIDATES>(
                                z1, col + 1,
                                local_top_values, local_top_indices);
                        }
                    }
                }
            }
            if ((vocab_size & 1) != 0 && tid == 0) {
                const int col = vocab_size - 1;
                const float z = logit_temperature *
                    decoded_logit_value<CENTERED_LOGITS>(
                        logits[row_offset + col], logit_centers, row, col,
                        logit_center_stride);
                local_max = fmaxf(local_max, z);
                if constexpr (OUTPUT_TOPK > 0) {
                    if (col != target) {
                        insert_topk<LOCAL_CANDIDATES>(
                            z, col, local_top_values, local_top_indices);
                    }
                }
            }
        } else {
            for (int col = tid; col < vocab_size; col += THREADS) {
                const float z = logit_temperature *
                    decoded_logit_value<CENTERED_LOGITS>(
                        logits[row_offset + col], logit_centers, row, col,
                        logit_center_stride);
                local_max = fmaxf(local_max, z);
                if constexpr (OUTPUT_TOPK > 0) {
                    if (col != target) {
                        insert_topk<LOCAL_CANDIDATES>(
                            z, col, local_top_values, local_top_indices);
                    }
                }
            }
        }
    }

    const float local_sum_reference_max = local_max;
    if constexpr (
        MXFP4_SINGLE_PASS &&
        MXFP4_TAIL_MATH_MODE == MXFP4_TAIL_NATIVE) {
        if (tid == 0) s_max = fixed_anchor;
    } else {
        local_max = warp_reduce_max(local_max);
        if (lane == 0) warp_red[warp] = local_max;
        __syncthreads();
        if (warp == 0) {
            float block_max = lane < ROW_WARPS ? warp_red[lane] : -INFINITY;
            block_max = warp_reduce_max(block_max);
            if (lane == 0) s_max = block_max;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int rank = 0; rank < WARP_CANDIDATES; ++rank) {
        float candidate_value = -INFINITY;
        int candidate_index = -1;
        #pragma unroll
        for (int local_rank = 0; local_rank < LOCAL_CANDIDATES; ++local_rank) {
            const int index = local_top_indices[local_rank];
            bool selected = false;
            #pragma unroll
            for (int prior = 0; prior < rank; ++prior) {
                if (index == warp_top_indices[warp][prior]) selected = true;
            }
            if (!selected && argmax_better(
                    local_top_values[local_rank], index,
                    candidate_value, candidate_index)) {
                candidate_value = local_top_values[local_rank];
                candidate_index = index;
            }
        }
        warp_reduce_argmax(candidate_value, candidate_index);
        if (lane == 0) {
            warp_top_values[warp][rank] = candidate_value;
            warp_top_indices[warp][rank] = candidate_index;
        }
        __syncwarp();
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int rank = 0; rank < OUTPUT_TOPK; ++rank) {
            float candidate_value = -INFINITY;
            int candidate_index = -1;
            if (lane < ROW_WARPS) {
                #pragma unroll
                for (int local_rank = 0; local_rank < WARP_CANDIDATES;
                     ++local_rank) {
                    const int index = warp_top_indices[lane][local_rank];
                    bool selected = false;
                    #pragma unroll
                    for (int prior = 0; prior < rank; ++prior) {
                        if (index == s_topk_indices[prior]) selected = true;
                    }
                    if (!selected && argmax_better(
                            warp_top_values[lane][local_rank], index,
                            candidate_value, candidate_index)) {
                        candidate_value = warp_top_values[lane][local_rank];
                        candidate_index = index;
                    }
                }
            }
            warp_reduce_argmax(candidate_value, candidate_index);
            if (lane == 0) s_topk_indices[rank] = candidate_index;
            __syncwarp();
        }
    }
    __syncthreads();

    // Remove exact-repair entries from the cached payload. Their numerators
    // still participate in the softmax denominator below.
    if (tid <= OUTPUT_TOPK) {
        const int selected_index = tid == 0
            ? target
            : s_topk_indices[tid - 1];
        if (is_valid && selected_index >= 0) {
            s_selected_logits[tid] = logit_temperature *
                decoded_logit_value<CENTERED_LOGITS>(
                    logits[row_offset + selected_index], logit_centers,
                    row, selected_index, logit_center_stride);
            if constexpr (MXFP4_SINGLE_PASS) {
                clear_mxfp4_row_value(
                    row_fp4, row, selected_index, N);
            }
            remove_logit(logits + row_offset + selected_index);
        } else {
            s_selected_logits[tid] = -INFINITY;
        }
    }
    __syncthreads();

    // The single-pass producer sees selected values before it knows which
    // entries exact repair will own. Requantize only the distinct 32-value
    // blocks touched by target/top-k so those large values do not inflate the
    // scale of the remaining tail. This is at most OUTPUT_TOPK + 1 tiny block
    // reads instead of a second full logits pass.
    if constexpr (MXFP4_SINGLE_PASS) {
        if (is_valid && tid <= OUTPUT_TOPK) {
            const int selected_index = tid == 0
                ? target
                : s_topk_indices[tid - 1];
            const int selected_block = selected_index / MXFP4_BLOCK;
            bool first_for_block = selected_index >= 0;
            #pragma unroll
            for (int prior = 0; prior < OUTPUT_TOPK + 1; ++prior) {
                if (prior < tid) {
                    const int prior_index = prior == 0
                        ? target
                        : s_topk_indices[prior - 1];
                    if (prior_index >= 0 &&
                        prior_index / MXFP4_BLOCK == selected_block) {
                        first_for_block = false;
                    }
                }
            }
            if (first_for_block) {
                quantize_mxfp4_exp_block<
                    FUSED_ROW_DATA_SR,
                    LOGIT_T,
                    false,
                    MXFP4_TAIL_MATH_MODE>(
                        logits,
                        row_fp4,
                        row_sc,
                        row,
                        selected_block,
                        N,
                        vocab_size,
                        fixed_anchor,
                        logit_temperature,
                        row_quant_parameter,
                        log2f(row_quant_parameter),
                        row_sr_nonce);
            }
        }
        __syncthreads();
    }

    if (!MXFP4_SINGLE_PASS && is_valid && tid <= OUTPUT_TOPK) {
        local_sum = __expf(s_selected_logits[tid] - s_max);
    }
    if constexpr (
        FUSED_ROW_FORMAT != FUSED_ROW_NONE && !MXFP4_SINGLE_PASS) {
        constexpr int FP4_BLOCK = FUSED_ROW_FORMAT == FUSED_ROW_NVFP4
            ? NVFP4_BLOCK
            : MXFP4_BLOCK;
        const int blocks = N / FP4_BLOCK;
        const float row_quant_log2_parameter =
            MXFP4_TAIL_MATH_MODE == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED
            ? log2f(row_quant_parameter)
            : 0.0f;
        for (int block = tid; block < blocks; block += THREADS) {
            if (is_valid) {
                if constexpr (FUSED_ROW_FORMAT == FUSED_ROW_NVFP4) {
                    local_sum += quantize_nvfp4_exp_block<FUSED_ROW_DATA_SR>(
                        logits, row_fp4, row_sc, row, block, N, vocab_size,
                        s_max, logit_temperature, row_quant_parameter,
                        row_sr_nonce);
                } else if constexpr (FUSED_ROW_FORMAT == FUSED_ROW_MXFP4) {
                    local_sum += quantize_mxfp4_exp_block<
                        FUSED_ROW_DATA_SR, LOGIT_T, CENTERED_LOGITS,
                        MXFP4_TAIL_MATH_MODE>(
                        logits, row_fp4, row_sc, row, block, N, vocab_size,
                        s_max, logit_temperature, row_quant_parameter,
                        row_quant_log2_parameter, row_sr_nonce,
                        logit_centers, logit_center_stride);
                } else {
                    local_sum += quantize_mxfp8_exp_block<FUSED_ROW_DATA_SR>(
                        logits, row_fp4, row_sc, row, block, N, vocab_size,
                        s_max, logit_temperature, row_quant_parameter,
                        row_sr_nonce);
                }
            } else {
                if constexpr (FUSED_ROW_FORMAT == FUSED_ROW_NVFP4) {
                    quantize_nvfp4_exp_block<FUSED_ROW_DATA_SR>(
                        logits, row_fp4, row_sc, row, block, N, 0,
                        0.0f, 1.0f, row_quant_parameter, row_sr_nonce);
                } else if constexpr (FUSED_ROW_FORMAT == FUSED_ROW_MXFP4) {
                    quantize_mxfp4_exp_block<
                        FUSED_ROW_DATA_SR, LOGIT_T, CENTERED_LOGITS,
                        MXFP4_TAIL_MATH_MODE>(
                        logits, row_fp4, row_sc, row, block, N, 0,
                        0.0f, 1.0f, row_quant_parameter,
                        row_quant_log2_parameter, row_sr_nonce, logit_centers,
                        logit_center_stride);
                } else {
                    quantize_mxfp8_exp_block<FUSED_ROW_DATA_SR>(
                        logits, row_fp4, row_sc, row, block, N, 0,
                        0.0f, 1.0f, row_quant_parameter, row_sr_nonce);
                }
            }
        }
    } else if constexpr (!MXFP4_SINGLE_PASS) {
        if (is_valid) {
            for (int col = tid; col < vocab_size; col += THREADS) {
                const float z = logit_temperature *
                    decoded_logit_value<CENTERED_LOGITS>(
                        logits[row_offset + col], logit_centers, row, col,
                        logit_center_stride);
                local_sum += __expf(z - s_max);
            }
        }
    }
    if constexpr (
        MXFP4_SINGLE_PASS &&
        MXFP4_TAIL_MATH_MODE ==
            MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
        local_sum *= __expf(local_sum_reference_max - s_max);
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = lane < ROW_WARPS ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            s_lse = sum > 0.0f
                ? s_max + logf(sum) +
                    ((MXFP4_SINGLE_PASS &&
                      MXFP4_TAIL_MATH_MODE ==
                          MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED)
                         ? fixed_anchor
                         : 0.0f)
                : 0.0f;
            lse[row] = s_lse;
            if constexpr (FUSED_ROW_FORMAT != FUSED_ROW_NONE) {
                row_max[row] = is_valid
                    ? (MXFP4_SINGLE_PASS ? fixed_anchor : s_max)
                    : 0.0f;
                if constexpr (FUSED_ROW_FORMAT == FUSED_ROW_NVFP4) {
                    if (row == 0) {
                        row_sg[0] = 1.0f / row_quant_parameter;
                    }
                }
            }
            if (is_valid) {
                const float target_logit = s_selected_logits[0];
                target_probs[row] = __expf(target_logit - s_lse);
                atomicAdd(loss_sum, s_lse - target_logit);
                atomicAdd(count_sum, 1.0f);
                #pragma unroll
                for (int rank = 0; rank < OUTPUT_TOPK; ++rank) {
                    const int index = s_topk_indices[rank];
                    const int64_t output =
                        static_cast<int64_t>(row) * OUTPUT_TOPK + rank;
                    if (index >= 0) {
                        const float z = s_selected_logits[rank + 1];
                        topk_probs[output] = __expf(z - s_lse);
                        topk_indices[output] = index;
                    } else {
                        topk_probs[output] = 0.0f;
                        topk_indices[output] = 0;
                    }
                }
            } else {
                target_probs[row] = 0.0f;
                #pragma unroll
                for (int rank = 0; rank < OUTPUT_TOPK; ++rank) {
                    const int64_t output =
                        static_cast<int64_t>(row) * OUTPUT_TOPK + rank;
                    topk_probs[output] = 0.0f;
                    topk_indices[output] = 0;
                }
            }
        }
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_split_next_top_kernel(
    __nv_bfloat16* __restrict__ logits,
    const bool* __restrict__ valid,
    const float* __restrict__ lse,
    float* __restrict__ topk_probs,
    int* __restrict__ topk_indices,
    int M,
    int N,
    int vocab_size,
    int topk_output_stride,
    int output_rank,
    float logit_temperature) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_values[ROW_WARPS];
    __shared__ int warp_indices[ROW_WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float local_value = -INFINITY;
    int local_index = -1;
    if (valid[row]) {
        const int64_t row_offset = static_cast<int64_t>(row) * N;
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float value = logit_temperature *
                __bfloat162float(logits[row_offset + col]);
            if (argmax_better(value, col, local_value, local_index)) {
                local_value = value;
                local_index = col;
            }
        }
    }
    warp_reduce_argmax(local_value, local_index);
    if (lane == 0) {
        warp_values[warp] = local_value;
        warp_indices[warp] = local_index;
    }
    __syncthreads();
    if (warp == 0) {
        float block_value =
            lane < ROW_WARPS ? warp_values[lane] : -INFINITY;
        int block_index = lane < ROW_WARPS ? warp_indices[lane] : -1;
        warp_reduce_argmax(block_value, block_index);
        if (lane == 0) {
            const int64_t output =
                static_cast<int64_t>(row) * topk_output_stride + output_rank;
            if (block_index >= 0) {
                topk_probs[output] = __expf(block_value - lse[row]);
                topk_indices[output] = block_index;
                logits[static_cast<int64_t>(row) * N + block_index] =
                    __float2bfloat16(-INFINITY);
            } else {
                topk_probs[output] = 0.0f;
                topk_indices[output] = 0;
            }
        }
    }
}

template <int THREADS, int TOPK_SPLIT, int EXACT_TOPK_SPLIT = TOPK_SPLIT>
__global__ __launch_bounds__(THREADS)
void exact_target_topk_loss_lse_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    float* __restrict__ target_probs,
    float* __restrict__ topk_probs,
    const int* __restrict__ topk_indices,
    float* __restrict__ corrected_loss_sum,
    float* __restrict__ corrected_count_sum,
    int M,
    int K,
    const float* __restrict__ row_max,
    float* __restrict__ row_normalization) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;
    if (!valid[row]) {
        if (tid == 0 && row_normalization != nullptr) {
            row_normalization[row] = 1.0f;
        }
        return;
    }

    static_assert(EXACT_TOPK_SPLIT >= 0 && EXACT_TOPK_SPLIT <= TOPK_SPLIT);
    constexpr int TERMS = EXACT_TOPK_SPLIT + 1;
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[TERMS][WARPS];
    __shared__ float exact_logits[TERMS];
    float local_sums[TERMS];
    #pragma unroll
    for (int term = 0; term < TERMS; ++term) {
        local_sums[term] = 0.0f;
    }

    const int64_t target = targets[row];
    const int64_t x_offset = static_cast<int64_t>(row) * K;
    for (int col = tid; col < K; col += THREADS) {
        const float x_value = __bfloat162float(x[x_offset + col]);
        local_sums[0] += x_value * __bfloat162float(
            weight[target * K + col]);
        #pragma unroll
        for (int rank = 0; rank < EXACT_TOPK_SPLIT; ++rank) {
            const int index = topk_indices[
                static_cast<int64_t>(row) * TOPK_SPLIT + rank];
            if (index >= 0) {
                local_sums[rank + 1] += x_value * __bfloat162float(
                    weight[static_cast<int64_t>(index) * K + col]);
            }
        }
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int term = 0; term < TERMS; ++term) {
        const float sum = warp_reduce_sum(local_sums[term]);
        if (lane == 0) warp_sums[term][warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            float sum = lane < WARPS ? warp_sums[term][lane] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane == 0) exact_logits[term] = sum;
        }
    }
    __syncthreads();

    if (tid == 0) {
        const float old_lse = lse[row];
        float tail_mass = 1.0f - target_probs[row];
        #pragma unroll
        for (int rank = 0; rank < EXACT_TOPK_SPLIT; ++rank) {
            tail_mass -= topk_probs[
                static_cast<int64_t>(row) * TOPK_SPLIT + rank];
        }
        tail_mass = fminf(fmaxf(tail_mass, 0.0f), 1.0f);

        float max_delta = 0.0f;
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            max_delta = fmaxf(max_delta, exact_logits[term] - old_lse);
        }
        float scaled_mass = tail_mass * __expf(-max_delta);
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            scaled_mass += __expf(
                exact_logits[term] - old_lse - max_delta);
        }
        const float new_lse = old_lse + max_delta + logf(
            fmaxf(scaled_mass, 1.0e-30f));
        const float approximate_probability_scale = __expf(old_lse - new_lse);
        lse[row] = new_lse;
        if (row_normalization != nullptr) {
            row_normalization[row] = __expf(row_max[row] - new_lse);
        }
        target_probs[row] = __expf(exact_logits[0] - new_lse);
        #pragma unroll
        for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
            const int64_t output =
                static_cast<int64_t>(row) * TOPK_SPLIT + rank;
            if (rank < EXACT_TOPK_SPLIT) {
                topk_probs[output] =
                    __expf(exact_logits[rank + 1] - new_lse);
            } else {
                topk_probs[output] *= approximate_probability_scale;
            }
        }
        atomicAdd(corrected_loss_sum, new_lse - exact_logits[0]);
        atomicAdd(corrected_count_sum, 1.0f);
    }
}

template <int THREADS, int TOPK_SPLIT, int EXACT_TOPK_SPLIT = TOPK_SPLIT>
__global__ __launch_bounds__(THREADS)
void exact_target_topk_loss_lse_vec8_kernel(
    const BFloat16x8* __restrict__ x,
    const BFloat16x8* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    float* __restrict__ target_probs,
    float* __restrict__ topk_probs,
    const int* __restrict__ topk_indices,
    float* __restrict__ corrected_loss_sum,
    float* __restrict__ corrected_count_sum,
    int M,
    int K8,
    const float* __restrict__ row_max,
    float* __restrict__ row_normalization) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;
    if (!valid[row]) {
        if (tid == 0 && row_normalization != nullptr) {
            row_normalization[row] = 1.0f;
        }
        return;
    }

    static_assert(EXACT_TOPK_SPLIT >= 0 && EXACT_TOPK_SPLIT <= TOPK_SPLIT);
    constexpr int TERMS = EXACT_TOPK_SPLIT + 1;
    constexpr int WARPS = THREADS / 32;
    __shared__ float warp_sums[TERMS][WARPS];
    __shared__ float exact_logits[TERMS];
    __shared__ int selected_rows[TERMS];
    if (tid == 0) selected_rows[0] = static_cast<int>(targets[row]);
    if (tid < EXACT_TOPK_SPLIT) {
        selected_rows[tid + 1] =
            topk_indices[static_cast<int64_t>(row) * TOPK_SPLIT + tid];
    }
    __syncthreads();

    float local_sums[TERMS];
    #pragma unroll
    for (int term = 0; term < TERMS; ++term) local_sums[term] = 0.0f;

    const int64_t x_offset = static_cast<int64_t>(row) * K8;
    for (int vector = tid; vector < K8; vector += THREADS) {
        float x_values[8];
        unpack_bfloat16x8(x[x_offset + vector], x_values);
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            const int selected = selected_rows[term];
            if (selected >= 0) {
                float weight_values[8];
                unpack_bfloat16x8(
                    weight[static_cast<int64_t>(selected) * K8 + vector],
                    weight_values);
                #pragma unroll
                for (int element = 0; element < 8; ++element) {
                    local_sums[term] = fmaf(
                        x_values[element], weight_values[element],
                        local_sums[term]);
                }
            }
        }
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    #pragma unroll
    for (int term = 0; term < TERMS; ++term) {
        const float sum = warp_reduce_sum(local_sums[term]);
        if (lane == 0) warp_sums[term][warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            float sum = lane < WARPS ? warp_sums[term][lane] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane == 0) exact_logits[term] = sum;
        }
    }
    __syncthreads();

    if (tid == 0) {
        const float old_lse = lse[row];
        float tail_mass = 1.0f - target_probs[row];
        #pragma unroll
        for (int rank = 0; rank < EXACT_TOPK_SPLIT; ++rank) {
            tail_mass -= topk_probs[
                static_cast<int64_t>(row) * TOPK_SPLIT + rank];
        }
        tail_mass = fminf(fmaxf(tail_mass, 0.0f), 1.0f);

        float max_delta = 0.0f;
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            max_delta = fmaxf(max_delta, exact_logits[term] - old_lse);
        }
        float scaled_mass = tail_mass * __expf(-max_delta);
        #pragma unroll
        for (int term = 0; term < TERMS; ++term) {
            scaled_mass += __expf(
                exact_logits[term] - old_lse - max_delta);
        }
        const float new_lse = old_lse + max_delta + logf(
            fmaxf(scaled_mass, 1.0e-30f));
        const float approximate_probability_scale = __expf(old_lse - new_lse);
        lse[row] = new_lse;
        if (row_normalization != nullptr) {
            row_normalization[row] = __expf(row_max[row] - new_lse);
        }
        target_probs[row] = __expf(exact_logits[0] - new_lse);
        #pragma unroll
        for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
            const int64_t output =
                static_cast<int64_t>(row) * TOPK_SPLIT + rank;
            if (rank < EXACT_TOPK_SPLIT) {
                topk_probs[output] =
                    __expf(exact_logits[rank + 1] - new_lse);
            } else {
                topk_probs[output] *= approximate_probability_scale;
            }
        }
        atomicAdd(corrected_loss_sum, new_lse - exact_logits[0]);
        atomicAdd(corrected_count_sum, 1.0f);
    }
}

template <int THREADS, int TOPK_SPLIT, int EXACT_TOPK_SPLIT>
void launch_exact_target_topk_loss_lse_threads(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const int64_t* targets,
    const bool* valid,
    float* lse,
    float* target_probs,
    float* topk_probs,
    const int* topk_indices,
    float* corrected_loss_sum,
    float* corrected_count_sum,
    int M,
    int K,
    cudaStream_t stream,
    const float* row_max,
    float* row_normalization) {
    if (K % 8 == 0) {
        exact_target_topk_loss_lse_vec8_kernel<
            THREADS, TOPK_SPLIT, EXACT_TOPK_SPLIT>
            <<<M, THREADS, 0, stream>>>(
                reinterpret_cast<const BFloat16x8*>(x),
                reinterpret_cast<const BFloat16x8*>(weight),
                targets, valid, lse, target_probs, topk_probs, topk_indices,
                corrected_loss_sum, corrected_count_sum, M, K / 8,
                row_max, row_normalization);
    } else {
        exact_target_topk_loss_lse_kernel<
            THREADS, TOPK_SPLIT, EXACT_TOPK_SPLIT>
            <<<M, THREADS, 0, stream>>>(
                x, weight, targets, valid, lse, target_probs, topk_probs,
                topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
                row_max, row_normalization);
    }
}

template <int TOPK_SPLIT, int EXACT_TOPK_SPLIT = TOPK_SPLIT>
void launch_exact_target_topk_loss_lse(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const int64_t* targets,
    const bool* valid,
    float* lse,
    float* target_probs,
    float* topk_probs,
    const int* topk_indices,
    float* corrected_loss_sum,
    float* corrected_count_sum,
    int M,
    int K,
    int threads,
    cudaStream_t stream,
    const float* row_max,
    float* row_normalization) {
#define LAUNCH_EXACT_SELECTED(THREADS)                                         \
    launch_exact_target_topk_loss_lse_threads<                                \
        THREADS, TOPK_SPLIT, EXACT_TOPK_SPLIT>(                               \
        x, weight, targets, valid, lse, target_probs, topk_probs,              \
        topk_indices, corrected_loss_sum, corrected_count_sum, M, K, stream,   \
        row_max, row_normalization)
    if (threads == 64) {
        LAUNCH_EXACT_SELECTED(64);
    } else if (threads == 128) {
        LAUNCH_EXACT_SELECTED(128);
    } else {
        LAUNCH_EXACT_SELECTED(256);
    }
#undef LAUNCH_EXACT_SELECTED
}

void dispatch_exact_target_topk_loss_lse(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const int64_t* targets,
    const bool* valid,
    float* lse,
    float* target_probs,
    float* topk_probs,
    const int* topk_indices,
    float* corrected_loss_sum,
    float* corrected_count_sum,
    int M,
    int K,
    int topk_split,
    cudaStream_t stream,
    const float* row_max = nullptr,
    float* row_normalization = nullptr) {
    const int threads = static_cast<int>(
        env_u64("FP4_CCE_V4_EXACT_SELECTED_THREADS", 128));
    const int exact_topk = static_cast<int>(env_u64(
        "FP4_CCE_V4_EXACT_SELECTED_TOPK", static_cast<uint64_t>(topk_split)));
    TORCH_CHECK(
        threads == 64 || threads == 128 || threads == 256,
        "FP4_CCE_V4_EXACT_SELECTED_THREADS must be 64, 128, or 256");
    TORCH_CHECK(
        exact_topk == topk_split ||
            (topk_split == 12 &&
             (exact_topk == 6 || exact_topk == 8)),
        "FP4_CCE_V4_EXACT_SELECTED_TOPK must equal the selected top-k, or "
        "be 6 or 8 when top-k is 12");
    if (topk_split == 16) {
        launch_exact_target_topk_loss_lse<16>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    } else if (topk_split == 12) {
        if (exact_topk == 6) {
            launch_exact_target_topk_loss_lse<12, 6>(
                x, weight, targets, valid, lse, target_probs, topk_probs,
                topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
                threads, stream, row_max, row_normalization);
        } else if (exact_topk == 8) {
            launch_exact_target_topk_loss_lse<12, 8>(
                x, weight, targets, valid, lse, target_probs, topk_probs,
                topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
                threads, stream, row_max, row_normalization);
        } else {
            launch_exact_target_topk_loss_lse<12>(
                x, weight, targets, valid, lse, target_probs, topk_probs,
                topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
                threads, stream, row_max, row_normalization);
        }
    } else if (topk_split == 8) {
        launch_exact_target_topk_loss_lse<8>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    } else if (topk_split == 6) {
        launch_exact_target_topk_loss_lse<6>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    } else if (topk_split == 4) {
        launch_exact_target_topk_loss_lse<4>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    } else if (topk_split == 2) {
        launch_exact_target_topk_loss_lse<2>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    } else if (topk_split == 1) {
        launch_exact_target_topk_loss_lse<1>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    } else {
        launch_exact_target_topk_loss_lse<0>(
            x, weight, targets, valid, lse, target_probs, topk_probs,
            topk_indices, corrected_loss_sum, corrected_count_sum, M, K,
            threads, stream, row_max, row_normalization);
    }
}

template <int THREADS, int TOPK_SPLIT, int TOPK_OUTPUT_STRIDE>
void launch_softmax_loss_lse_target_topk_split(
    __nv_bfloat16* logits,
    const int64_t* targets,
    const bool* valid,
    float* lse,
    float* target_probs,
    float* topk_probs,
    int* topk_indices,
    float* loss_sum,
    float* count_sum,
    int M,
    int N,
    int vocab_size,
    float logit_temperature,
    cudaStream_t stream) {
    softmax_loss_lse_target_topk_split_kernel<
        THREADS, TOPK_SPLIT, TOPK_OUTPUT_STRIDE>
        <<<M, THREADS, 0, stream>>>(
            logits,
            targets,
            valid,
            lse,
            target_probs,
            topk_probs,
            topk_indices,
            loss_sum,
            count_sum,
            M,
            N,
            vocab_size,
            logit_temperature);
}

template <int THREADS>
void dispatch_softmax_loss_lse_target_topk_split(
    __nv_bfloat16* logits,
    const int64_t* targets,
    const bool* valid,
    float* lse,
    float* target_probs,
    float* topk_probs,
    int* topk_indices,
    float* loss_sum,
    float* count_sum,
    int M,
    int N,
    int vocab_size,
    int topk_split,
    float logit_temperature,
    cudaStream_t stream) {
    if (topk_split == 16) {
        softmax_loss_lse_target_hierarchical_topk_kernel<THREADS, 16>
            <<<M, THREADS, 0, stream>>>(
                logits, targets, valid, lse, target_probs, topk_probs,
                topk_indices, loss_sum, count_sum, M, N, vocab_size,
                logit_temperature, nullptr, nullptr, nullptr, nullptr, 1.0f,
                0.0f);
    } else if (topk_split == 12) {
        softmax_loss_lse_target_hierarchical_topk_kernel<THREADS, 12>
            <<<M, THREADS, 0, stream>>>(
                logits, targets, valid, lse, target_probs, topk_probs,
                topk_indices, loss_sum, count_sum, M, N, vocab_size,
                logit_temperature, nullptr, nullptr, nullptr, nullptr, 1.0f,
                0.0f);
    } else if (topk_split == 8) {
        softmax_loss_lse_target_hierarchical_topk_kernel<THREADS, 8>
            <<<M, THREADS, 0, stream>>>(
                logits, targets, valid, lse, target_probs, topk_probs,
                topk_indices, loss_sum, count_sum, M, N, vocab_size,
                logit_temperature, nullptr, nullptr, nullptr, nullptr, 1.0f,
                0.0f);
    } else if (topk_split == 6) {
        launch_softmax_loss_lse_target_topk_split<THREADS, 1, 6>(
            logits, targets, valid, lse, target_probs, topk_probs,
            topk_indices, loss_sum, count_sum, M, N, vocab_size,
            logit_temperature, stream);
        for (int rank = 1; rank < 6; ++rank) {
            softmax_split_next_top_kernel<256><<<M, 256, 0, stream>>>(
                logits,
                valid,
                lse,
                topk_probs,
                topk_indices,
                M,
                N,
                vocab_size,
                6,
                rank,
                logit_temperature);
        }
    } else if (topk_split == 4) {
        launch_softmax_loss_lse_target_topk_split<THREADS, 4, 4>(
            logits, targets, valid, lse, target_probs, topk_probs,
            topk_indices, loss_sum, count_sum, M, N, vocab_size,
            logit_temperature, stream);
    } else if (topk_split == 2) {
        launch_softmax_loss_lse_target_topk_split<THREADS, 1, 2>(
            logits, targets, valid, lse, target_probs, topk_probs,
            topk_indices, loss_sum, count_sum, M, N, vocab_size,
            logit_temperature, stream);
        softmax_split_next_top_kernel<256><<<M, 256, 0, stream>>>(
            logits,
            valid,
            lse,
            topk_probs,
            topk_indices,
            M,
            N,
            vocab_size,
            2,
            1,
            logit_temperature);
    } else if (topk_split == 1) {
        launch_softmax_loss_lse_target_topk_split<THREADS, 1, 1>(
            logits, targets, valid, lse, target_probs, topk_probs,
            topk_indices, loss_sum, count_sum, M, N, vocab_size,
            logit_temperature, stream);
    } else {
        launch_softmax_loss_lse_target_topk_split<THREADS, 0, 0>(
            logits, targets, valid, lse, target_probs, topk_probs,
            topk_indices, loss_sum, count_sum, M, N, vocab_size,
            logit_temperature, stream);
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_lse_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    int M,
    int N,
    int vocab_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_red[ROW_WARPS];
    __shared__ float s_max;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bool is_valid = valid[row];

    float local_max = -INFINITY;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            local_max = fmaxf(local_max, __bfloat162float(logits[static_cast<int64_t>(row) * N + col]));
        }
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
            local_sum += __expf(z - s_max);
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS) ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            lse[row] = (sum > 0.0f) ? (s_max + logf(sum)) : 0.0f;
        }
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_lse_targets_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    float* __restrict__ lse,
    float* __restrict__ target_logits,
    int64_t* __restrict__ local_targets_out,
    int M,
    int N,
    int vocab_start,
    int global_vocab_size,
    int vocab_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int ROW_WARPS = THREADS / 32;
    __shared__ float warp_red[ROW_WARPS];
    __shared__ float s_max;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const bool is_valid = valid[row];
    const int64_t target = targets[row];
    const int64_t local_target = target - static_cast<int64_t>(vocab_start);
    const bool in_range = (
        is_valid &&
        local_target >= 0 &&
        local_target < static_cast<int64_t>(vocab_size) &&
        target < static_cast<int64_t>(global_vocab_size)
    );

    if (tid == 0) {
        local_targets_out[row] = in_range ? local_target : static_cast<int64_t>(-1);
        target_logits[row] = in_range
            ? __bfloat162float(logits[static_cast<int64_t>(row) * N + local_target])
            : -INFINITY;
    }

    float local_max = -INFINITY;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            local_max = fmaxf(local_max, __bfloat162float(logits[static_cast<int64_t>(row) * N + col]));
        }
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_red[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float block_max = (lane < ROW_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    if (is_valid) {
        for (int col = tid; col < vocab_size; col += THREADS) {
            const float z = __bfloat162float(logits[static_cast<int64_t>(row) * N + col]);
            local_sum += __expf(z - s_max);
        }
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_red[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < ROW_WARPS) ? warp_red[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            lse[row] = (sum > 0.0f) ? (s_max + logf(sum)) : 0.0f;
        }
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_grad_probs_from_lse_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const int64_t* __restrict__ local_targets,
    const bool* __restrict__ valid,
    __nv_bfloat16* __restrict__ grad_probs,
    int M,
    int N,
    int vocab_size,
    float logit_temperature) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    const int64_t row_offset = static_cast<int64_t>(row) * N;
    const bool is_valid = valid[row];
    const int64_t target = local_targets[row];
    const float row_lse = lse[row];
    for (int col = tid; col < N; col += THREADS) {
        float p = 0.0f;
        if (is_valid && col < vocab_size) {
            const float z = logit_temperature *
                __bfloat162float(logits[row_offset + col]);
            p = __expf(z - row_lse);
            if (col == target) {
                p -= 1.0f;
            }
        }
        grad_probs[row_offset + col] = __float2bfloat16(p);
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_tail_grad_probs_no_target_from_lse_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    __nv_bfloat16* __restrict__ grad_probs,
    int M,
    int N,
    int vocab_size,
    float logit_temperature) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    const int64_t logits_offset = static_cast<int64_t>(row) * N;
    const int64_t output_offset = static_cast<int64_t>(row) * N;
    const bool is_valid = valid[row];
    const float row_lse = lse[row];
    for (int col = tid; col < N; col += THREADS) {
        float p = 0.0f;
        if (is_valid && col < vocab_size) {
            const float z = logit_temperature *
                __bfloat162float(logits[logits_offset + col]);
            p = __expf(z - row_lse);
        }
        grad_probs[output_offset + col] = __float2bfloat16(p);
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS)
void softmax_tail_grad_probs_no_target_from_lse_inplace_kernel(
    __nv_bfloat16* logits_and_grad,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    int M,
    int N,
    int vocab_size,
    float logit_temperature) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    const int64_t row_offset = static_cast<int64_t>(row) * N;
    const bool is_valid = valid[row];
    const float row_lse = lse[row];
    for (int col = tid; col < N; col += THREADS) {
        float p = 0.0f;
        if (is_valid && col < vocab_size) {
            // Load before overwriting.  This dedicated single-pointer kernel
            // deliberately avoids aliasing the restrict-qualified input and
            // output arguments of the out-of-place producer.
            const float z = logit_temperature *
                __bfloat162float(logits_and_grad[row_offset + col]);
            p = __expf(z - row_lse);
        }
        logits_and_grad[row_offset + col] = __float2bfloat16(p);
    }
}

__global__ void scatter_exact_selected_grad_probs_kernel(
    __nv_bfloat16* __restrict__ grad_probs,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    const float* __restrict__ target_probs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_probs,
    int M,
    int N,
    int topk) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int terms = topk + 1;
    if (item >= M * terms) return;
    const int row = item / terms;
    const int term = item - row * terms;
    if (!valid[row]) return;

    const int col = term == 0
        ? static_cast<int>(targets[row])
        : topk_indices[static_cast<int64_t>(row) * topk + term - 1];
    if (col < 0 || col >= N) return;
    const float value = term == 0
        ? target_probs[row] - 1.0f
        : topk_probs[static_cast<int64_t>(row) * topk + term - 1];
    grad_probs[static_cast<int64_t>(row) * N + col] =
        __float2bfloat16(value);
}

void replace_target_logits_bf16(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    int64_t ignore_index,
    int64_t vocab_size) {
    TORCH_CHECK(
        logits.is_cuda() && x.is_cuda() && weight.is_cuda() && targets.is_cuda(),
        "logits, x, weight, and targets must be CUDA tensors");
    TORCH_CHECK(
        logits.is_contiguous() && x.is_contiguous() && weight.is_contiguous() &&
            targets.is_contiguous(),
        "logits, x, weight, and targets must be contiguous");
    TORCH_CHECK(
        logits.scalar_type() == torch::kBFloat16 &&
            x.scalar_type() == torch::kBFloat16 &&
            weight.scalar_type() == torch::kBFloat16,
        "logits, x, and weight must be BF16");
    TORCH_CHECK(
        targets.scalar_type() == torch::kInt64,
        "targets must be int64");
    TORCH_CHECK(
        logits.dim() == 2 && x.dim() == 2 && weight.dim() == 2,
        "logits, x, and weight must be rank-2");
    TORCH_CHECK(
        logits.size(0) == x.size(0) && targets.numel() == x.size(0),
        "row count mismatch");
    TORCH_CHECK(x.size(1) == weight.size(1), "hidden dimension mismatch");
    TORCH_CHECK(
        vocab_size >= 0 && vocab_size <= logits.size(1) &&
            vocab_size <= weight.size(0),
        "invalid vocab_size");

    constexpr int THREADS = 256;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    replace_target_logits_bf16_kernel<THREADS>
        <<<x.size(0), THREADS, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            static_cast<int>(x.size(0)),
            static_cast<int>(logits.size(1)),
            static_cast<int>(x.size(1)),
            static_cast<int>(vocab_size),
            ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_probs_impl(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    bool target_split,
    int topk_split) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        topk_split == 0 || topk_split == 1 || topk_split == 2 ||
            topk_split == 4,
        "top-k split must be 0, 1, 2, or 4");

    auto device = logits.device();
    auto probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto target_probs = target_split
        ? torch::empty({M64}, torch::dtype(torch::kFloat32).device(device))
        : torch::empty({0}, torch::dtype(torch::kFloat32).device(device));
    float* target_probs_ptr =
        target_split ? target_probs.data_ptr<float>() : nullptr;
    auto topk_probs = topk_split == 1
        ? torch::empty({M64}, torch::dtype(torch::kFloat32).device(device))
        : (topk_split > 1
            ? torch::empty(
                {M64, topk_split},
                torch::dtype(torch::kFloat32).device(device))
            : torch::empty({0}, torch::dtype(torch::kFloat32).device(device)));
    auto topk_indices = topk_split == 1
        ? torch::empty({M64}, torch::dtype(torch::kInt32).device(device))
        : (topk_split > 1
            ? torch::empty(
                {M64, topk_split},
                torch::dtype(torch::kInt32).device(device))
            : torch::empty({0}, torch::dtype(torch::kInt32).device(device)));
    float* topk_probs_ptr =
        topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr;
    int* topk_indices_ptr =
        topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr;
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    const bool stage_exp = use_stage_exp(std::getenv("FP4_CCE_V4_SOFTMAX_STAGE_EXP"));
    if (selected_threads == 1024 && stage_exp) {
        softmax_loss_probs_kernel<1024, true><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            target_probs_ptr,
            topk_probs_ptr,
            topk_indices_ptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            target_split,
            topk_split);
    } else if (selected_threads == 1024) {
        softmax_loss_probs_kernel<1024, false><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            target_probs_ptr,
            topk_probs_ptr,
            topk_indices_ptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            target_split,
            topk_split);
    } else if (selected_threads == 512 && stage_exp) {
        softmax_loss_probs_kernel<512, true><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            target_probs_ptr,
            topk_probs_ptr,
            topk_indices_ptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            target_split,
            topk_split);
    } else if (selected_threads == 512) {
        softmax_loss_probs_kernel<512, false><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            target_probs_ptr,
            topk_probs_ptr,
            topk_indices_ptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            target_split,
            topk_split);
    } else if (stage_exp) {
        softmax_loss_probs_kernel<ROW_THREADS, true><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            target_probs_ptr,
            topk_probs_ptr,
            topk_indices_ptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            target_split,
            topk_split);
    } else {
        softmax_loss_probs_kernel<ROW_THREADS, false><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            target_probs_ptr,
            topk_probs_ptr,
            topk_indices_ptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            false,
            target_split,
            topk_split);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(
        loss.reshape({}), probs, target_probs, topk_probs, topk_indices);
}

std::tuple<torch::Tensor, torch::Tensor>
softmax_loss_probs(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    auto result = softmax_loss_probs_impl(
        logits, targets, valid, vocab_size, false, 0);
    return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
softmax_loss_probs_target_split(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    auto result = softmax_loss_probs_impl(
        logits, targets, valid, vocab_size, true, 0);
    return std::make_tuple(
        std::get<0>(result), std::get<1>(result), std::get<2>(result));
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_probs_target_top1_split(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    return softmax_loss_probs_impl(
        logits, targets, valid, vocab_size, true, 1);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_probs_target_top2_split(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    return softmax_loss_probs_impl(
        logits, targets, valid, vocab_size, true, 2);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_probs_target_top4_split(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    return softmax_loss_probs_impl(
        logits, targets, valid, vocab_size, true, 4);
}

std::tuple<torch::Tensor, torch::Tensor>
softmax_loss_grad_probs(torch::Tensor logits, torch::Tensor targets, torch::Tensor valid, int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto grad_probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    const bool stage_exp = use_stage_exp(std::getenv("FP4_CCE_V4_SOFTMAX_STAGE_EXP"));
    if (selected_threads == 1024 && stage_exp) {
        softmax_loss_probs_kernel<1024, true><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            false,
            false);
    } else if (selected_threads == 1024) {
        softmax_loss_probs_kernel<1024, false><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            false,
            false);
    } else if (selected_threads == 512 && stage_exp) {
        softmax_loss_probs_kernel<512, true><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            false,
            false);
    } else if (selected_threads == 512) {
        softmax_loss_probs_kernel<512, false><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            false,
            false);
    } else if (stage_exp) {
        softmax_loss_probs_kernel<ROW_THREADS, true><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            false,
            false);
    } else {
        softmax_loss_probs_kernel<ROW_THREADS, false><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            nullptr,
            nullptr,
            nullptr,
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            true,
            false,
            false);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(loss.reshape({}), grad_probs);
}

std::tuple<torch::Tensor, torch::Tensor>
softmax_loss_grad_probs_all_valid_full_vocab(
    torch::Tensor logits,
    torch::Tensor targets,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(targets.numel() == M64, "target length mismatch");
    TORCH_CHECK(vocab_size == N64, "all-valid/full-vocab fast path requires vocab_size == logits.size(1)");

    auto device = logits.device();
    auto grad_probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    const bool stage_exp = use_stage_exp(std::getenv("FP4_CCE_V4_SOFTMAX_STAGE_EXP"));
    if (selected_threads == 1024 && stage_exp) {
        softmax_loss_grad_probs_all_valid_kernel<1024, true><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            loss_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64));
    } else if (selected_threads == 1024) {
        softmax_loss_grad_probs_all_valid_kernel<1024, false><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            loss_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64));
    } else if (selected_threads == 512 && stage_exp) {
        softmax_loss_grad_probs_all_valid_kernel<512, true><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            loss_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64));
    } else if (selected_threads == 512) {
        softmax_loss_grad_probs_all_valid_kernel<512, false><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            loss_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64));
    } else if (stage_exp) {
        softmax_loss_grad_probs_all_valid_kernel<ROW_THREADS, true><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            loss_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64));
    } else {
        softmax_loss_grad_probs_all_valid_kernel<ROW_THREADS, false><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            loss_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / static_cast<double>(M64 > 0 ? M64 : 1);
    return std::make_tuple(loss.reshape({}), grad_probs);
}

std::tuple<torch::Tensor, torch::Tensor>
softmax_loss_lse(torch::Tensor logits, torch::Tensor targets, torch::Tensor valid, int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    if (selected_threads == 1024) {
        softmax_loss_lse_kernel<1024><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size));
    } else if (selected_threads == 512) {
        softmax_loss_lse_kernel<512><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size));
    } else {
        softmax_loss_lse_kernel<ROW_THREADS><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(loss.reshape({}), lse);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_lse_target_topk_split(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        topk_split == 0 || topk_split == 1 || topk_split == 2 ||
            topk_split == 4 || topk_split == 6 || topk_split == 8 ||
            topk_split == 12 || topk_split == 16,
        "top-k split must be 0, 1, 2, 4, 6, 8, 12, or 16");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto target_probs = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto topk_probs = topk_split == 1
        ? torch::empty({M64}, torch::dtype(torch::kFloat32).device(device))
        : (topk_split > 1
            ? torch::empty(
                {M64, topk_split},
                torch::dtype(torch::kFloat32).device(device))
            : torch::empty({0}, torch::dtype(torch::kFloat32).device(device)));
    auto topk_indices = topk_split == 1
        ? torch::empty({M64}, torch::dtype(torch::kInt32).device(device))
        : (topk_split > 1
            ? torch::empty(
                {M64, topk_split},
                torch::dtype(torch::kInt32).device(device))
            : torch::empty({0}, torch::dtype(torch::kInt32).device(device)));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const float logit_temperature = mxfp8_logit_temperature();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads =
        row_threads == nullptr && topk_split >= 8
        ? 256
        : select_row_threads(row_threads, vocab_size);
    float* topk_probs_ptr =
        topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr;
    int* topk_indices_ptr =
        topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr;
    auto logits_ptr = reinterpret_cast<__nv_bfloat16*>(logits.data_ptr());
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int V = static_cast<int>(vocab_size);
    if (selected_threads == 1024) {
        dispatch_softmax_loss_lse_target_topk_split<1024>(
            logits_ptr, targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),
            topk_probs_ptr, topk_indices_ptr, loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(), M, N, V,
            static_cast<int>(topk_split), logit_temperature, stream);
    } else if (selected_threads == 512) {
        dispatch_softmax_loss_lse_target_topk_split<512>(
            logits_ptr, targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),
            topk_probs_ptr, topk_indices_ptr, loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(), M, N, V,
            static_cast<int>(topk_split), logit_temperature, stream);
    } else {
        dispatch_softmax_loss_lse_target_topk_split<ROW_THREADS>(
            logits_ptr, targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),
            topk_probs_ptr, topk_indices_ptr, loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(), M, N, V,
            static_cast<int>(topk_split), logit_temperature, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(
        loss.reshape({}), lse, target_probs, topk_probs, topk_indices);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_lse_target_topk_split_exact_logits(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split) {
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == torch::kBFloat16 && x.dim() == 2,
        "x must be contiguous CUDA BF16 [M, K]");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
            weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2,
        "weight must be contiguous CUDA BF16 [V, K]");
    TORCH_CHECK(
        logits.device() == x.device() && x.device() == weight.device() &&
            weight.device() == targets.device() && targets.device() == valid.device(),
        "exact selected-logit inputs must be on one CUDA device");
    TORCH_CHECK(
        x.size(0) == logits.size(0) && x.size(1) == weight.size(1),
        "exact selected-logit shape mismatch");
    TORCH_CHECK(
        weight.size(0) >= vocab_size,
        "weight must contain every vocabulary row");
    TORCH_CHECK(
        topk_split == 1 || topk_split == 2 || topk_split == 4 ||
            topk_split == 6 || topk_split == 8 || topk_split == 12 ||
            topk_split == 16,
        "exact selected-logit repair requires top-k split 1, 2, 4, 6, 8, 12, or 16");

    auto outputs = softmax_loss_lse_target_topk_split(
        logits, targets, valid, vocab_size, topk_split);
    auto lse = std::get<1>(outputs);
    auto target_probs = std::get<2>(outputs);
    auto topk_probs = std::get<3>(outputs);
    auto topk_indices = std::get<4>(outputs);
    auto corrected_loss_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(logits.device()));
    auto corrected_count_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(logits.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dispatch_exact_target_topk_loss_lse(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        targets.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        lse.data_ptr<float>(),
        target_probs.data_ptr<float>(),
        topk_probs.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        corrected_loss_sum.data_ptr<float>(),
        corrected_count_sum.data_ptr<float>(),
        static_cast<int>(x.size(0)),
        static_cast<int>(x.size(1)),
        static_cast<int>(topk_split),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = corrected_loss_sum / torch::clamp_min(corrected_count_sum, 1.0);
    return std::make_tuple(
        loss.reshape({}), lse, target_probs, topk_probs, topk_indices);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_lse_target_topk_split_exact_logits_nvfp4_row(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double global_scale_max,
    bool exact_selected_logits) {
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2,
        "logits must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == torch::kBFloat16 && x.dim() == 2,
        "x must be contiguous CUDA BF16 [M, K]");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
            weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2,
        "weight must be contiguous CUDA BF16 [V, K]");
    TORCH_CHECK(
        targets.is_cuda() && targets.is_contiguous() &&
            targets.scalar_type() == torch::kInt64,
        "targets must be contiguous CUDA int64");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");
    TORCH_CHECK(
        logits.device() == x.device() && x.device() == weight.device() &&
            weight.device() == targets.device() && targets.device() == valid.device(),
        "fused NVFP4 row inputs must be on one CUDA device");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        M64 % NVFP4_TILE == 0 && N64 % NVFP4_TILE == 0,
        "fused NVFP4 row dimensions must be multiples of 128");
    TORCH_CHECK(
        targets.numel() == M64 && valid.numel() == M64,
        "target/valid length mismatch");
    TORCH_CHECK(
        x.size(0) == M64 && x.size(1) == weight.size(1),
        "exact selected-logit shape mismatch");
    TORCH_CHECK(
        vocab_size >= 0 && vocab_size <= N64 && weight.size(0) >= vocab_size,
        "invalid vocabulary size");
    TORCH_CHECK(
        topk_split == 0 || topk_split == 8 || topk_split == 12 ||
            topk_split == 16,
        "fused NVFP4 row producer requires top-k 0, 8, 12, or 16");
    TORCH_CHECK(
        std::isfinite(global_scale_max) && global_scale_max > 0.0 &&
            global_scale_max <= static_cast<double>(FLT_MAX / 6.0f),
        "global_scale_max must be finite, positive, and representable");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto target_probs = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto topk_probs = torch::empty(
        {M64, topk_split}, torch::dtype(torch::kFloat32).device(device));
    auto topk_indices = torch::empty(
        {M64, topk_split}, torch::dtype(torch::kInt32).device(device));
    auto row_fp4 = torch::empty(
        {M64, N64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M64 / NVFP4_TILE, N64 / 64, 512},
        torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto row_sg = torch::empty(
        {1}, torch::dtype(torch::kFloat32).device(device));
    auto row_max = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto row_normalization = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto loss_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor corrected_loss_sum;
    torch::Tensor corrected_count_sum;
    if (exact_selected_logits) {
        corrected_loss_sum = torch::zeros(
            {1}, torch::dtype(torch::kFloat32).device(device));
        corrected_count_sum = torch::zeros(
            {1}, torch::dtype(torch::kFloat32).device(device));
    }

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int V = static_cast<int>(vocab_size);
    const float logit_temperature = mxfp8_logit_temperature();
    const float global_scale_denom = static_cast<float>(global_scale_max * 6.0);
    const bool row_data_sr =
        env_flag("FP4_CCE_V4_NVFP4_DATA_SR", false) ||
        env_flag("FP4_CCE_V4_NVFP4_USE_STOCHASTIC_ROUNDING", false) ||
        env_flag("FP4_CCE_V4_NVFP4_G_ROW_DATA_SR", false);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto row_rng_state = row_data_sr
        ? make_fused_row_rng_state(
              logits,
              env_u64("FP4_CCE_V4_NVFP4_RNG_SEED", 0) ^
                  (env_u64("RANK", 0) * 0xd1b54a32d192ed03ull),
              env_u64("FP4_CCE_V4_NVFP4_RNG_SUBSEQUENCE_BASE", 0),
              stream)
        : torch::Tensor();
    const auto* row_rng_state_ptr = row_rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(
              row_rng_state.data_ptr<int64_t>())
        : nullptr;
#define LAUNCH_FUSED_NVFP4_ROW(TOPK, ROW_SR)                                   \
    softmax_loss_lse_target_hierarchical_topk_kernel<                           \
        256, TOPK, FUSED_ROW_NVFP4, ROW_SR>                                     \
        <<<M, 256, 0, stream>>>(                                                \
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),               \
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),                \
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),              \
            topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr,            \
            topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr,            \
            loss_sum.data_ptr<float>(), count_sum.data_ptr<float>(),            \
            M, N, V, logit_temperature,                                         \
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),             \
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),                      \
            row_max.data_ptr<float>(), row_sg.data_ptr<float>(),                \
            global_scale_denom, 0.0f, row_rng_state_ptr)
#define DISPATCH_FUSED_NVFP4_ROW(TOPK)                                         \
    do {                                                                        \
        if (row_data_sr) {                                                      \
            LAUNCH_FUSED_NVFP4_ROW(TOPK, true);                                 \
        } else {                                                                \
            LAUNCH_FUSED_NVFP4_ROW(TOPK, false);                                \
        }                                                                       \
    } while (false)
    if (topk_split == 16) {
        DISPATCH_FUSED_NVFP4_ROW(16);
    } else if (topk_split == 12) {
        DISPATCH_FUSED_NVFP4_ROW(12);
    } else if (topk_split == 8) {
        DISPATCH_FUSED_NVFP4_ROW(8);
    } else {
        DISPATCH_FUSED_NVFP4_ROW(0);
    }
#undef DISPATCH_FUSED_NVFP4_ROW
#undef LAUNCH_FUSED_NVFP4_ROW

    if (exact_selected_logits) {
        dispatch_exact_target_topk_loss_lse(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),
            topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr,
            topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr,
            corrected_loss_sum.data_ptr<float>(),
            corrected_count_sum.data_ptr<float>(), M,
            static_cast<int>(x.size(1)), static_cast<int>(topk_split), stream,
            row_max.data_ptr<float>(), row_normalization.data_ptr<float>());
    }
    // Keep the row payload in numerator space. Backward applies this row-wise
    // denominator once to dE and folds it into the transposed dW operand.
    if (!exact_selected_logits) {
        make_nvfp4_row_normalization_kernel<<<
            std::min((M + 255) / 256, 128), 256, 0, stream>>>(
                row_max.data_ptr<float>(), lse.data_ptr<float>(),
                row_normalization.data_ptr<float>(), M);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = exact_selected_logits
        ? corrected_loss_sum / torch::clamp_min(corrected_count_sum, 1.0)
        : loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(
        loss.reshape({}), lse, target_probs, topk_probs, topk_indices,
        row_fp4, row_sc, row_sg, row_normalization);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_impl(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double scale_floor_ratio,
    bool exact_selected_logits,
    torch::Tensor logit_centers);

auto softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double scale_floor_ratio,
    bool exact_selected_logits) {
    return softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_impl(
        logits, x, weight, targets, valid, vocab_size, topk_split,
        scale_floor_ratio, exact_selected_logits, torch::Tensor());
}

auto softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered(
    torch::Tensor logit_residuals,
    torch::Tensor logit_centers,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double scale_floor_ratio,
    bool exact_selected_logits) {
    return softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_impl(
        logit_residuals, x, weight, targets, valid, vocab_size, topk_split,
        scale_floor_ratio, exact_selected_logits, logit_centers);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_impl(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double scale_floor_ratio,
    bool exact_selected_logits,
    torch::Tensor logit_centers) {
    const bool centered_logits =
        logit_centers.defined() && logit_centers.numel() > 0;
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            (logits.scalar_type() == torch::kBFloat16 ||
             logits.scalar_type() == torch::kFloat8_e4m3fn) &&
            logits.dim() == 2,
        "logits must be contiguous CUDA BF16 or E4M3 [M, N]");
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == torch::kBFloat16 && x.dim() == 2,
        "x must be contiguous CUDA BF16 [M, K]");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
            weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2,
        "weight must be contiguous CUDA BF16 [V, K]");
    TORCH_CHECK(
        targets.is_cuda() && targets.is_contiguous() &&
            targets.scalar_type() == torch::kInt64,
        "targets must be contiguous CUDA int64");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");
    TORCH_CHECK(
        logits.device() == x.device() && x.device() == weight.device() &&
            weight.device() == targets.device() && targets.device() == valid.device(),
        "fused MXFP4 row inputs must be on one CUDA device");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        M64 % NVFP4_TILE == 0 && N64 % NVFP4_TILE == 0,
        "fused MXFP4 row dimensions must be multiples of 128");
    if (centered_logits) {
        TORCH_CHECK(
            logits.scalar_type() == torch::kFloat8_e4m3fn,
            "centered logits require an E4M3 residual tensor");
        TORCH_CHECK(
            logit_centers.is_cuda() && logit_centers.is_contiguous() &&
                logit_centers.scalar_type() == torch::kBFloat16 &&
                logit_centers.dim() == 2 &&
                logit_centers.size(0) == M64 &&
                logit_centers.size(1) * 32 == N64,
            "logit centers must be contiguous CUDA BF16 [M, N/32]");
        TORCH_CHECK(
            logit_centers.device() == logits.device(),
            "logit centers and residuals must be on one CUDA device");
    }
    TORCH_CHECK(
        targets.numel() == M64 && valid.numel() == M64,
        "target/valid length mismatch");
    TORCH_CHECK(
        x.size(0) == M64 && x.size(1) == weight.size(1),
        "exact selected-logit shape mismatch");
    TORCH_CHECK(
        vocab_size >= 0 && vocab_size <= N64 && weight.size(0) >= vocab_size,
        "invalid vocabulary size");
    TORCH_CHECK(
        topk_split == 0 || topk_split == 8 || topk_split == 12 ||
            topk_split == 16,
        "fused MXFP4 row producer requires top-k 0, 8, 12, or 16");
    TORCH_CHECK(
        std::isfinite(scale_floor_ratio) && scale_floor_ratio >= 1.0 &&
            scale_floor_ratio <= 2.0,
        "scale_floor_ratio must be finite and in [1, 2]");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto target_probs = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto topk_probs = torch::empty(
        {M64, topk_split}, torch::dtype(torch::kFloat32).device(device));
    auto topk_indices = torch::empty(
        {M64, topk_split}, torch::dtype(torch::kInt32).device(device));
    auto row_fp4 = torch::empty(
        {M64, N64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M64 / NVFP4_TILE, N64 / NVFP4_TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto row_sg = torch::empty(
        {0}, torch::dtype(torch::kFloat32).device(device));
    auto row_max = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto row_normalization = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto loss_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor corrected_loss_sum;
    torch::Tensor corrected_count_sum;
    if (exact_selected_logits) {
        corrected_loss_sum = torch::zeros(
            {1}, torch::dtype(torch::kFloat32).device(device));
        corrected_count_sum = torch::zeros(
            {1}, torch::dtype(torch::kFloat32).device(device));
    }

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int V = static_cast<int>(vocab_size);
    const float logit_temperature = mxfp8_logit_temperature();
    const bool fp8_logits = logits.scalar_type() == torch::kFloat8_e4m3fn;
    const bool row_data_sr =
        env_flag("FP4_CCE_V4_MXFP4_G_ROW_DATA_SR", false);
    const int row_threads = static_cast<int>(
        env_u64("FP4_CCE_V4_MXFP4_G_ROW_THREADS", 128));
    const int tail_math = mxfp4_tail_math_mode();
    const bool single_pass =
        env_flag("FP4_CCE_V4_MXFP4_G_SINGLE_PASS", false);
    const bool fast_top12 =
        env_flag("FP4_CCE_V4_MXFP4_G_FAST_TOP12", false);
    const bool mid_top12 =
        env_flag("FP4_CCE_V4_MXFP4_G_MID_TOP12", false);
    const float fixed_anchor = mxfp4_single_pass_anchor();
    TORCH_CHECK(
        row_threads == 64 || row_threads == 128 || row_threads == 256,
        "FP4_CCE_V4_MXFP4_G_ROW_THREADS must be 64, 128, or 256");
    TORCH_CHECK(
        !single_pass ||
            (!fp8_logits && !centered_logits && row_threads == 128 &&
             (tail_math == MXFP4_TAIL_NATIVE ||
              tail_math == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) &&
             exact_selected_logits &&
             (topk_split == 8 || topk_split == 12 || topk_split == 16)),
        "single-pass MXFP4 G-cache requires native or log-code BF16 logits, 128 row "
        "threads, exact selected logits, and top-k 8, 12, or 16");
    TORCH_CHECK(
        !fast_top12 ||
            (!fp8_logits && !centered_logits && row_threads == 128 &&
             tail_math == MXFP4_TAIL_NATIVE && !single_pass &&
             topk_split == 12),
        "fast MXFP4 top-12 requires native BF16 logits, 128 row threads, "
        "and the two-pass top-12 producer");
    TORCH_CHECK(
        !(fast_top12 && mid_top12),
        "fast and mid MXFP4 top-12 selection are mutually exclusive");
    TORCH_CHECK(
        !mid_top12 ||
            (!fp8_logits && !centered_logits && row_threads == 128 &&
             tail_math == MXFP4_TAIL_NATIVE && !single_pass &&
             topk_split == 12),
        "mid MXFP4 top-12 requires native BF16 logits, 128 row threads, "
        "and the two-pass top-12 producer");
    const auto* logit_center_ptr = centered_logits
        ? reinterpret_cast<const __nv_bfloat16*>(logit_centers.data_ptr())
        : nullptr;
    const int logit_center_stride = centered_logits
        ? static_cast<int>(logit_centers.size(1))
        : 0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto row_rng_state = row_data_sr
        ? make_fused_row_rng_state(
              logits,
              env_u64("FP4_CCE_V4_MXFP4_G_RNG_SEED", 0) ^
                  (env_u64("RANK", 0) * 0xd1b54a32d192ed03ull),
              env_u64("FP4_CCE_V4_MXFP4_G_RNG_SUBSEQUENCE_BASE", 0),
              stream)
        : torch::Tensor();
    const auto* row_rng_state_ptr = row_rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(
              row_rng_state.data_ptr<int64_t>())
        : nullptr;
#define LAUNCH_FUSED_MXFP4_ROW(                                                \
    THREADS, TOPK, ROW_SR, LOGIT_T, CENTERED, TAIL_MATH, SINGLE_PASS)          \
    softmax_loss_lse_target_hierarchical_topk_kernel<                           \
        THREADS, TOPK, FUSED_ROW_MXFP4, ROW_SR, LOGIT_T, CENTERED, TAIL_MATH,   \
        SINGLE_PASS>                                                           \
        <<<M, THREADS, 0, stream>>>(                                            \
            reinterpret_cast<LOGIT_T*>(logits.data_ptr()),                     \
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),                \
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),              \
            topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr,            \
            topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr,            \
            loss_sum.data_ptr<float>(), count_sum.data_ptr<float>(),            \
            M, N, V, logit_temperature,                                         \
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),             \
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),                      \
            row_max.data_ptr<float>(), nullptr,                                 \
            static_cast<float>(scale_floor_ratio), fixed_anchor,                \
            row_rng_state_ptr,                                                  \
            logit_center_ptr, logit_center_stride)
#define DISPATCH_FUSED_MXFP4_ROW_THREADS(THREADS, TOPK)                        \
    do {                                                                        \
        if (fp8_logits) {                                                       \
            if (centered_logits && row_data_sr) {                               \
                LAUNCH_FUSED_MXFP4_ROW(                                         \
                    THREADS, TOPK, true, __nv_fp8_e4m3, true,                   \
                    MXFP4_TAIL_NATIVE, false);                                  \
            } else if (centered_logits) {                                       \
                LAUNCH_FUSED_MXFP4_ROW(                                         \
                    THREADS, TOPK, false, __nv_fp8_e4m3, true,                  \
                    MXFP4_TAIL_NATIVE, false);                                  \
            } else if (row_data_sr) {                                           \
                LAUNCH_FUSED_MXFP4_ROW(                                         \
                    THREADS, TOPK, true, __nv_fp8_e4m3, false,                  \
                    MXFP4_TAIL_NATIVE, false);                                  \
            } else {                                                            \
                LAUNCH_FUSED_MXFP4_ROW(                                         \
                    THREADS, TOPK, false, __nv_fp8_e4m3, false,                 \
                    MXFP4_TAIL_NATIVE, false);                                  \
            }                                                                   \
        } else if (row_data_sr) {                                               \
            LAUNCH_FUSED_MXFP4_ROW(                                             \
                THREADS, TOPK, true, __nv_bfloat16, false,                      \
                MXFP4_TAIL_NATIVE, false);                                      \
        } else {                                                                \
            LAUNCH_FUSED_MXFP4_ROW(                                             \
                THREADS, TOPK, false, __nv_bfloat16, false,                     \
                MXFP4_TAIL_NATIVE, false);                                      \
        }                                                                       \
    } while (false)
#define DISPATCH_FUSED_MXFP4_ROW(TOPK)                                         \
    do {                                                                        \
        if (row_threads == 256) {                                               \
            DISPATCH_FUSED_MXFP4_ROW_THREADS(256, TOPK);                        \
        } else if (row_threads == 64) {                                         \
            DISPATCH_FUSED_MXFP4_ROW_THREADS(64, TOPK);                         \
        } else {                                                                \
            DISPATCH_FUSED_MXFP4_ROW_THREADS(128, TOPK);                        \
        }                                                                       \
    } while (false)
#define DISPATCH_EXPERIMENTAL_MXFP4_TAIL(TAIL_MATH)                            \
    do {                                                                        \
        if (row_data_sr) {                                                      \
            LAUNCH_FUSED_MXFP4_ROW(                                             \
                128, 12, true, __nv_bfloat16, false, TAIL_MATH, false);         \
        } else {                                                                \
            LAUNCH_FUSED_MXFP4_ROW(                                             \
                128, 12, false, __nv_bfloat16, false, TAIL_MATH, false);        \
        }                                                                       \
    } while (false)
#define DISPATCH_FAST_MXFP4_TOP12(ROW_SR)                                     \
    softmax_loss_lse_target_hierarchical_topk_kernel<                          \
        128, 12, FUSED_ROW_MXFP4, ROW_SR, __nv_bfloat16, false,               \
        MXFP4_TAIL_NATIVE, false, true>                                        \
        <<<M, 128, 0, stream>>>(                                               \
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),              \
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),               \
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),             \
            topk_probs.data_ptr<float>(), topk_indices.data_ptr<int>(),        \
            loss_sum.data_ptr<float>(), count_sum.data_ptr<float>(),           \
            M, N, V, logit_temperature,                                        \
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),            \
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),                     \
            row_max.data_ptr<float>(), nullptr,                                \
            static_cast<float>(scale_floor_ratio), 0.0f,                       \
            row_rng_state_ptr, nullptr, 0)
#define DISPATCH_MID_MXFP4_TOP12(ROW_SR)                                      \
    softmax_loss_lse_target_hierarchical_topk_kernel<                          \
        128, 12, FUSED_ROW_MXFP4, ROW_SR, __nv_bfloat16, false,               \
        MXFP4_TAIL_NATIVE, false, false, true>                                 \
        <<<M, 128, 0, stream>>>(                                               \
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),              \
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),               \
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),             \
            topk_probs.data_ptr<float>(), topk_indices.data_ptr<int>(),        \
            loss_sum.data_ptr<float>(), count_sum.data_ptr<float>(),           \
            M, N, V, logit_temperature,                                        \
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),            \
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),                     \
            row_max.data_ptr<float>(), nullptr,                                \
            static_cast<float>(scale_floor_ratio), 0.0f,                       \
            row_rng_state_ptr, nullptr, 0)
#define DISPATCH_SINGLE_PASS_MXFP4_ROW(TOPK, TAIL_MATH)                       \
    do {                                                                        \
        if (row_data_sr) {                                                      \
            LAUNCH_FUSED_MXFP4_ROW(                                             \
                128, TOPK, true, __nv_bfloat16, false,                         \
                TAIL_MATH, true);                                               \
        } else {                                                                \
            LAUNCH_FUSED_MXFP4_ROW(                                             \
                128, TOPK, false, __nv_bfloat16, false,                        \
                TAIL_MATH, true);                                               \
        }                                                                       \
    } while (false)
    if (fast_top12) {
        if (row_data_sr) {
            DISPATCH_FAST_MXFP4_TOP12(true);
        } else {
            DISPATCH_FAST_MXFP4_TOP12(false);
        }
    } else if (mid_top12) {
        if (row_data_sr) {
            DISPATCH_MID_MXFP4_TOP12(true);
        } else {
            DISPATCH_MID_MXFP4_TOP12(false);
        }
    } else if (single_pass) {
        if (tail_math == MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED) {
            if (topk_split == 16) {
                DISPATCH_SINGLE_PASS_MXFP4_ROW(
                    16, MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED);
            } else if (topk_split == 12) {
                DISPATCH_SINGLE_PASS_MXFP4_ROW(
                    12, MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED);
            } else {
                DISPATCH_SINGLE_PASS_MXFP4_ROW(
                    8, MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED);
            }
        } else if (topk_split == 16) {
            DISPATCH_SINGLE_PASS_MXFP4_ROW(16, MXFP4_TAIL_NATIVE);
        } else if (topk_split == 12) {
            DISPATCH_SINGLE_PASS_MXFP4_ROW(12, MXFP4_TAIL_NATIVE);
        } else {
            DISPATCH_SINGLE_PASS_MXFP4_ROW(8, MXFP4_TAIL_NATIVE);
        }
    } else if (tail_math != MXFP4_TAIL_NATIVE) {
        TORCH_CHECK(
            !fp8_logits && !centered_logits && row_threads == 128 &&
                topk_split == 12,
            "experimental MXFP4 tail math currently requires BF16 logits, "
            "128 row threads, and top-12 repair");
        if (tail_math == MXFP4_TAIL_POLY4) {
            DISPATCH_EXPERIMENTAL_MXFP4_TAIL(MXFP4_TAIL_POLY4);
        } else if (tail_math == MXFP4_TAIL_POLY4_REPRESENTED) {
            DISPATCH_EXPERIMENTAL_MXFP4_TAIL(
                MXFP4_TAIL_POLY4_REPRESENTED);
        } else if (tail_math == MXFP4_TAIL_AFFINE_REPRESENTED) {
            DISPATCH_EXPERIMENTAL_MXFP4_TAIL(
                MXFP4_TAIL_AFFINE_REPRESENTED);
        } else if (tail_math == MXFP4_TAIL_HYBRID25) {
            DISPATCH_EXPERIMENTAL_MXFP4_TAIL(MXFP4_TAIL_HYBRID25);
        } else if (tail_math == MXFP4_TAIL_HYBRID25_REPRESENTED) {
            DISPATCH_EXPERIMENTAL_MXFP4_TAIL(
                MXFP4_TAIL_HYBRID25_REPRESENTED);
        } else {
            DISPATCH_EXPERIMENTAL_MXFP4_TAIL(
                MXFP4_TAIL_LOGCODE_EXACT_REPRESENTED);
        }
    } else if (topk_split == 16) {
        DISPATCH_FUSED_MXFP4_ROW(16);
    } else if (topk_split == 12) {
        DISPATCH_FUSED_MXFP4_ROW(12);
    } else if (topk_split == 8) {
        DISPATCH_FUSED_MXFP4_ROW(8);
    } else {
        DISPATCH_FUSED_MXFP4_ROW(0);
    }
#undef DISPATCH_SINGLE_PASS_MXFP4_ROW
#undef DISPATCH_MID_MXFP4_TOP12
#undef DISPATCH_FAST_MXFP4_TOP12
#undef DISPATCH_EXPERIMENTAL_MXFP4_TAIL
#undef DISPATCH_FUSED_MXFP4_ROW
#undef DISPATCH_FUSED_MXFP4_ROW_THREADS
#undef LAUNCH_FUSED_MXFP4_ROW

    if (exact_selected_logits) {
        dispatch_exact_target_topk_loss_lse(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),
            topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr,
            topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr,
            corrected_loss_sum.data_ptr<float>(),
            corrected_count_sum.data_ptr<float>(), M,
            static_cast<int>(x.size(1)), static_cast<int>(topk_split), stream,
            row_max.data_ptr<float>(), row_normalization.data_ptr<float>());
    }
    if (!exact_selected_logits) {
        make_nvfp4_row_normalization_kernel<<<
            std::min((M + 255) / 256, 128), 256, 0, stream>>>(
                row_max.data_ptr<float>(), lse.data_ptr<float>(),
                row_normalization.data_ptr<float>(), M);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = exact_selected_logits
        ? corrected_loss_sum / torch::clamp_min(corrected_count_sum, 1.0)
        : loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(
        loss.reshape({}), lse, target_probs, topk_probs, topk_indices,
        row_fp4, row_sc, row_sg, row_normalization);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row_with_sr_state(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double quant_max,
    bool exact_selected_logits,
    torch::Tensor persistent_row_sr_state) {
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2,
        "logits must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == torch::kBFloat16 && x.dim() == 2,
        "x must be contiguous CUDA BF16 [M, K]");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
            weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2,
        "weight must be contiguous CUDA BF16 [V, K]");
    TORCH_CHECK(
        targets.is_cuda() && targets.is_contiguous() &&
            targets.scalar_type() == torch::kInt64,
        "targets must be contiguous CUDA int64");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");
    TORCH_CHECK(
        logits.device() == x.device() && x.device() == weight.device() &&
            weight.device() == targets.device() && targets.device() == valid.device(),
        "fused MXFP8 row inputs must be on one CUDA device");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        M64 % NVFP4_TILE == 0 && N64 % NVFP4_TILE == 0,
        "fused MXFP8 row dimensions must be multiples of 128");
    TORCH_CHECK(
        targets.numel() == M64 && valid.numel() == M64,
        "target/valid length mismatch");
    TORCH_CHECK(
        x.size(0) == M64 && x.size(1) == weight.size(1),
        "exact selected-logit shape mismatch");
    TORCH_CHECK(
        vocab_size >= 0 && vocab_size <= N64 && weight.size(0) >= vocab_size,
        "invalid vocabulary size");
    TORCH_CHECK(
        topk_split == 0 || topk_split == 8 || topk_split == 12 ||
            topk_split == 16,
        "fused MXFP8 row producer requires top-k 0, 8, 12, or 16");
    TORCH_CHECK(
        std::isfinite(quant_max) && quant_max > 0.0,
        "quant_max must be finite and positive");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto target_probs = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto topk_probs = torch::empty(
        {M64, topk_split}, torch::dtype(torch::kFloat32).device(device));
    auto topk_indices = torch::empty(
        {M64, topk_split}, torch::dtype(torch::kInt32).device(device));
    auto row_fp8 = torch::empty(
        {M64, N64}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    auto row_sc = torch::empty(
        {M64 / NVFP4_TILE, N64 / NVFP4_TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto row_sg = torch::empty(
        {0}, torch::dtype(torch::kFloat32).device(device));
    auto row_max = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto row_normalization = torch::empty(
        {M64}, torch::dtype(torch::kFloat32).device(device));
    auto loss_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor corrected_loss_sum;
    torch::Tensor corrected_count_sum;
    if (exact_selected_logits) {
        corrected_loss_sum = torch::zeros(
            {1}, torch::dtype(torch::kFloat32).device(device));
        corrected_count_sum = torch::zeros(
            {1}, torch::dtype(torch::kFloat32).device(device));
    }

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int V = static_cast<int>(vocab_size);
    const float logit_temperature = mxfp8_logit_temperature();
    const bool row_data_sr =
        env_flag("FP4_CCE_V4_NVFP4_DATA_SR", false) ||
        env_flag("FP4_CCE_V4_NVFP4_USE_STOCHASTIC_ROUNDING", false) ||
        env_flag("FP4_CCE_V4_NVFP4_G_ROW_DATA_SR", false);
    const bool checkpointed_row_sr =
        env_flag("FP4_CCE_V4_CHECKPOINTED_HEAD_SR", false);
    TORCH_CHECK(
        !checkpointed_row_sr || row_data_sr,
        "FP4_CCE_V4_CHECKPOINTED_HEAD_SR requires fused output-head row SR");
    TORCH_CHECK(
        !checkpointed_row_sr || persistent_row_sr_state.defined(),
        "checkpointed fused output-head row SR state was not installed");
    TORCH_CHECK(
        checkpointed_row_sr || !persistent_row_sr_state.defined(),
        "persistent fused output-head row SR state requires "
        "FP4_CCE_V4_CHECKPOINTED_HEAD_SR=1");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto row_rng_state = row_data_sr
        ? make_fused_row_rng_state(
              logits,
              env_u64(
                  "FP4_CCE_V4_MXFP8_G_RNG_SEED",
                  env_u64("FP4_CCE_V4_NVFP4_RNG_SEED", 0)) ^
                  (env_u64("RANK", 0) * 0xd1b54a32d192ed03ull),
              env_u64(
                  "FP4_CCE_V4_MXFP8_G_RNG_SUBSEQUENCE_BASE",
                  env_u64("FP4_CCE_V4_NVFP4_RNG_SUBSEQUENCE_BASE", 0)),
              stream,
              persistent_row_sr_state)
        : torch::Tensor();
    const auto* row_rng_state_ptr = row_rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(
              row_rng_state.data_ptr<int64_t>())
        : nullptr;
#define LAUNCH_FUSED_MXFP8_ROW(TOPK, ROW_SR)                                   \
    softmax_loss_lse_target_hierarchical_topk_kernel<                           \
        256, TOPK, FUSED_ROW_MXFP8, ROW_SR>                                     \
        <<<M, 256, 0, stream>>>(                                                \
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),               \
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),                \
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),              \
            topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr,            \
            topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr,            \
            loss_sum.data_ptr<float>(), count_sum.data_ptr<float>(),            \
            M, N, V, logit_temperature,                                         \
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp8.data_ptr()),             \
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),                      \
            row_max.data_ptr<float>(), nullptr, static_cast<float>(quant_max),  \
            0.0f, row_rng_state_ptr)
#define DISPATCH_FUSED_MXFP8_ROW(TOPK)                                         \
    do {                                                                        \
        if (row_data_sr) {                                                      \
            LAUNCH_FUSED_MXFP8_ROW(TOPK, true);                                 \
        } else {                                                                \
            LAUNCH_FUSED_MXFP8_ROW(TOPK, false);                                \
        }                                                                       \
    } while (false)
    if (topk_split == 16) {
        DISPATCH_FUSED_MXFP8_ROW(16);
    } else if (topk_split == 12) {
        DISPATCH_FUSED_MXFP8_ROW(12);
    } else if (topk_split == 8) {
        DISPATCH_FUSED_MXFP8_ROW(8);
    } else {
        DISPATCH_FUSED_MXFP8_ROW(0);
    }
#undef DISPATCH_FUSED_MXFP8_ROW
#undef LAUNCH_FUSED_MXFP8_ROW

    if (exact_selected_logits) {
        dispatch_exact_target_topk_loss_lse(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(), valid.data_ptr<bool>(),
            lse.data_ptr<float>(), target_probs.data_ptr<float>(),
            topk_split > 0 ? topk_probs.data_ptr<float>() : nullptr,
            topk_split > 0 ? topk_indices.data_ptr<int>() : nullptr,
            corrected_loss_sum.data_ptr<float>(),
            corrected_count_sum.data_ptr<float>(), M,
            static_cast<int>(x.size(1)), static_cast<int>(topk_split), stream,
            row_max.data_ptr<float>(), row_normalization.data_ptr<float>());
    }
    if (!exact_selected_logits) {
        make_nvfp4_row_normalization_kernel<<<
            std::min((M + 255) / 256, 128), 256, 0, stream>>>(
                row_max.data_ptr<float>(), lse.data_ptr<float>(),
                row_normalization.data_ptr<float>(), M);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto loss = exact_selected_logits
        ? corrected_loss_sum / torch::clamp_min(corrected_count_sum, 1.0)
        : loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(
        loss.reshape({}), lse, target_probs, topk_probs, topk_indices,
        row_fp8, row_sc, row_sg, row_normalization);
}

auto softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row(
    torch::Tensor logits,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    int64_t topk_split,
    double quant_max,
    bool exact_selected_logits) {
    return softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row_with_sr_state(
        logits, x, weight, targets, valid, vocab_size, topk_split, quant_max,
        exact_selected_logits, torch::Tensor());
}

torch::Tensor softmax_lse(torch::Tensor logits, torch::Tensor valid, int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(valid.numel() == M64, "valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    if (selected_threads == 1024) {
        softmax_lse_kernel<1024><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size));
    } else if (selected_threads == 512) {
        softmax_lse_kernel<512><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size));
    } else {
        softmax_lse_kernel<ROW_THREADS><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return lse;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
softmax_lse_targets(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_start,
    int64_t global_vocab_size,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64,
                "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size > 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(vocab_start >= 0 && vocab_start <= global_vocab_size, "invalid vocab_start");

    auto device = logits.device();
    auto lse = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto target_logits = torch::empty({M64}, torch::dtype(torch::kFloat32).device(device));
    auto local_targets = torch::empty({M64}, torch::dtype(torch::kInt64).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    if (selected_threads == 1024) {
        softmax_lse_targets_kernel<1024><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            target_logits.data_ptr<float>(),
            local_targets.data_ptr<int64_t>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_start),
            static_cast<int>(global_vocab_size),
            static_cast<int>(vocab_size));
    } else if (selected_threads == 512) {
        softmax_lse_targets_kernel<512><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            target_logits.data_ptr<float>(),
            local_targets.data_ptr<int64_t>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_start),
            static_cast<int>(global_vocab_size),
            static_cast<int>(vocab_size));
    } else {
        softmax_lse_targets_kernel<ROW_THREADS><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            lse.data_ptr<float>(),
            target_logits.data_ptr<float>(),
            local_targets.data_ptr<int64_t>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_start),
            static_cast<int>(global_vocab_size),
            static_cast<int>(vocab_size));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(lse, target_logits, local_targets);
}

torch::Tensor softmax_grad_probs_from_lse_impl(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor local_targets,
    torch::Tensor valid,
    int64_t vocab_size,
    double logit_temperature) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat,
                "lse must be contiguous CUDA float");
    TORCH_CHECK(local_targets.is_cuda() && local_targets.is_contiguous() && local_targets.scalar_type() == torch::kInt64,
                "local_targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool,
                "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(lse.numel() == M64 && local_targets.numel() == M64 && valid.numel() == M64,
                "lse/target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        std::isfinite(logit_temperature) && logit_temperature > 0.0,
        "logit_temperature must be finite and positive");

    auto device = logits.device();
    auto grad_probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
    if (selected_threads == 1024) {
        softmax_grad_probs_from_lse_kernel<1024><<<M64, 1024, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            local_targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            static_cast<float>(logit_temperature));
    } else if (selected_threads == 512) {
        softmax_grad_probs_from_lse_kernel<512><<<M64, 512, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            local_targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            static_cast<float>(logit_temperature));
    } else {
        softmax_grad_probs_from_lse_kernel<ROW_THREADS><<<M64, ROW_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            local_targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            static_cast<float>(logit_temperature));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return grad_probs;
}

torch::Tensor softmax_grad_probs_from_lse(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor local_targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    return softmax_grad_probs_from_lse_impl(
        logits, lse, local_targets, valid, vocab_size, 1.0);
}

torch::Tensor softmax_tail_grad_probs_from_lse(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    double logit_temperature) {
    auto no_local_target = torch::full(
        {logits.size(0)},
        -1,
        torch::dtype(torch::kInt64).device(logits.device()));
    return softmax_grad_probs_from_lse_impl(
        logits,
        lse,
        no_local_target,
        valid,
        vocab_size,
        logit_temperature);
}

torch::Tensor softmax_tail_grad_probs_no_target_from_lse(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    double logit_temperature) {
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
        logits.device() == lse.device() && logits.device() == valid.device(),
        "logits/lse/valid device mismatch");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        lse.numel() == M64 && valid.numel() == M64,
        "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        std::isfinite(logit_temperature) && logit_temperature > 0.0,
        "logit_temperature must be finite and positive");

    auto grad_probs = torch::empty(
        {M64, N64},
        torch::dtype(torch::kBFloat16).device(logits.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
#define LAUNCH_TAIL(THREADS)                                                    \
    softmax_tail_grad_probs_no_target_from_lse_kernel<THREADS>                 \
        <<<M64, THREADS, 0, stream>>>(                                          \
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),         \
            lse.data_ptr<float>(),                                              \
            valid.data_ptr<bool>(),                                             \
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),           \
            static_cast<int>(M64),                                              \
            static_cast<int>(N64),                                              \
            static_cast<int>(vocab_size),                                       \
            static_cast<float>(logit_temperature))
    if (selected_threads == 1024) {
        LAUNCH_TAIL(1024);
    } else if (selected_threads == 512) {
        LAUNCH_TAIL(512);
    } else {
        LAUNCH_TAIL(ROW_THREADS);
    }
#undef LAUNCH_TAIL
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_probs;
}

torch::Tensor softmax_repaired_grad_probs_from_lse(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor targets,
    torch::Tensor valid,
    torch::Tensor target_probs,
    torch::Tensor topk_indices,
    torch::Tensor topk_probs,
    int64_t vocab_size,
    double logit_temperature) {
    TORCH_CHECK(
        targets.is_cuda() && targets.is_contiguous() &&
            targets.scalar_type() == torch::kInt64,
        "targets must be contiguous CUDA int64");
    TORCH_CHECK(
        target_probs.is_cuda() && target_probs.is_contiguous() &&
            target_probs.scalar_type() == torch::kFloat,
        "target_probs must be contiguous CUDA float");
    TORCH_CHECK(
        topk_indices.is_cuda() && topk_indices.is_contiguous() &&
            topk_indices.scalar_type() == torch::kInt32 &&
            topk_indices.dim() == 2,
        "topk_indices must be contiguous CUDA int32 [M, topk]");
    TORCH_CHECK(
        topk_probs.is_cuda() && topk_probs.is_contiguous() &&
            topk_probs.scalar_type() == torch::kFloat &&
            topk_probs.dim() == 2,
        "topk_probs must be contiguous CUDA float [M, topk]");
    TORCH_CHECK(
        targets.device() == logits.device() &&
            target_probs.device() == logits.device() &&
            topk_indices.device() == logits.device() &&
            topk_probs.device() == logits.device(),
        "selected repair device mismatch");
    const int64_t M64 = logits.size(0);
    TORCH_CHECK(
        targets.numel() == M64 && target_probs.numel() == M64 &&
            topk_indices.size(0) == M64 && topk_probs.size(0) == M64 &&
            topk_indices.sizes() == topk_probs.sizes(),
        "selected repair shape mismatch");

    auto grad_probs = softmax_tail_grad_probs_no_target_from_lse(
        logits,
        lse,
        valid,
        vocab_size,
        logit_temperature);
    const int topk = static_cast<int>(topk_indices.size(1));
    const int64_t items = M64 * (topk + 1);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((items + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    scatter_exact_selected_grad_probs_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
        targets.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        target_probs.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        topk_probs.data_ptr<float>(),
        static_cast<int>(M64),
        static_cast<int>(logits.size(1)),
        topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_probs;
}

torch::Tensor softmax_repaired_grad_probs_from_lse_inplace(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor targets,
    torch::Tensor valid,
    torch::Tensor target_probs,
    torch::Tensor topk_indices,
    torch::Tensor topk_probs,
    int64_t vocab_size,
    double logit_temperature) {
    TORCH_CHECK(
        logits.is_cuda() && logits.is_contiguous() &&
            logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2,
        "logits must be contiguous CUDA BF16 [M, N]");
    TORCH_CHECK(
        lse.is_cuda() && lse.is_contiguous() &&
            lse.scalar_type() == torch::kFloat,
        "lse must be contiguous CUDA float");
    TORCH_CHECK(
        targets.is_cuda() && targets.is_contiguous() &&
            targets.scalar_type() == torch::kInt64,
        "targets must be contiguous CUDA int64");
    TORCH_CHECK(
        valid.is_cuda() && valid.is_contiguous() &&
            valid.scalar_type() == torch::kBool,
        "valid must be contiguous CUDA bool");
    TORCH_CHECK(
        target_probs.is_cuda() && target_probs.is_contiguous() &&
            target_probs.scalar_type() == torch::kFloat,
        "target_probs must be contiguous CUDA float");
    TORCH_CHECK(
        topk_indices.is_cuda() && topk_indices.is_contiguous() &&
            topk_indices.scalar_type() == torch::kInt32 &&
            topk_indices.dim() == 2,
        "topk_indices must be contiguous CUDA int32 [M, topk]");
    TORCH_CHECK(
        topk_probs.is_cuda() && topk_probs.is_contiguous() &&
            topk_probs.scalar_type() == torch::kFloat &&
            topk_probs.dim() == 2,
        "topk_probs must be contiguous CUDA float [M, topk]");
    TORCH_CHECK(
        logits.device() == lse.device() &&
            logits.device() == targets.device() &&
            logits.device() == valid.device() &&
            logits.device() == target_probs.device() &&
            logits.device() == topk_indices.device() &&
            logits.device() == topk_probs.device(),
        "repaired gradient input device mismatch");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(
        lse.numel() == M64 && targets.numel() == M64 &&
            valid.numel() == M64 && target_probs.numel() == M64 &&
            topk_indices.size(0) == M64 && topk_probs.size(0) == M64 &&
            topk_indices.sizes() == topk_probs.sizes(),
        "repaired gradient input shape mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        std::isfinite(logit_temperature) && logit_temperature > 0.0,
        "logit_temperature must be finite and positive");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* row_threads = std::getenv("FP4_CCE_V4_SOFTMAX_ROW_THREADS");
    const int selected_threads = select_row_threads(row_threads, vocab_size);
#define LAUNCH_TAIL_INPLACE(THREADS)                                            \
    softmax_tail_grad_probs_no_target_from_lse_inplace_kernel<THREADS>         \
        <<<M64, THREADS, 0, stream>>>(                                          \
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),               \
            lse.data_ptr<float>(),                                              \
            valid.data_ptr<bool>(),                                             \
            static_cast<int>(M64),                                              \
            static_cast<int>(N64),                                              \
            static_cast<int>(vocab_size),                                       \
            static_cast<float>(logit_temperature))
    if (selected_threads == 1024) {
        LAUNCH_TAIL_INPLACE(1024);
    } else if (selected_threads == 512) {
        LAUNCH_TAIL_INPLACE(512);
    } else {
        LAUNCH_TAIL_INPLACE(ROW_THREADS);
    }
#undef LAUNCH_TAIL_INPLACE
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int topk = static_cast<int>(topk_indices.size(1));
    const int64_t items = M64 * (topk + 1);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((items + threads - 1) / threads);
    scatter_exact_selected_grad_probs_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()),
        targets.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        target_probs.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        topk_probs.data_ptr<float>(),
        static_cast<int>(M64),
        static_cast<int>(N64),
        topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return logits;
}

torch::Tensor bf16_logits_cuda(torch::Tensor hidden, torch::Tensor weight) {
    TORCH_CHECK(
        hidden.is_cuda() && hidden.is_contiguous() &&
            hidden.scalar_type() == torch::kBFloat16 && hidden.dim() == 2,
        "hidden must be contiguous CUDA BF16 [M, K]");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
            weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2,
        "weight must be contiguous CUDA BF16 [V, K]");
    TORCH_CHECK(hidden.device() == weight.device(), "hidden/weight device mismatch");
    TORCH_CHECK(hidden.size(1) == weight.size(1), "hidden/weight K mismatch");

    const int M = static_cast<int>(hidden.size(0));
    const int K = static_cast<int>(hidden.size(1));
    const int V = static_cast<int>(weight.size(0));
    auto logits = torch::empty(
        {hidden.size(0), weight.size(0)},
        torch::dtype(torch::kBFloat16).device(hidden.device()));

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        V,
        M,
        K,
        &alpha,
        weight.data_ptr(),
        CUDA_R_16BF,
        K,
        hidden.data_ptr(),
        CUDA_R_16BF,
        K,
        &beta,
        logits.data_ptr(),
        CUDA_R_16BF,
        V,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return logits;
}

std::tuple<torch::Tensor, torch::Tensor> bf16_tail_grads_cuda(
    torch::Tensor grad_logits,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor grad_output,
    torch::Tensor valid_count) {
    TORCH_CHECK(
        grad_logits.is_cuda() && grad_logits.is_contiguous() &&
            grad_logits.scalar_type() == torch::kBFloat16 &&
            grad_logits.dim() == 2,
        "grad_logits must be contiguous CUDA BF16 [M, V]");
    TORCH_CHECK(
        hidden.is_cuda() && hidden.is_contiguous() &&
            hidden.scalar_type() == torch::kBFloat16 && hidden.dim() == 2,
        "hidden must be contiguous CUDA BF16 [M, K]");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
            weight.scalar_type() == torch::kBFloat16 && weight.dim() == 2,
        "weight must be contiguous CUDA BF16 [V, K]");
    TORCH_CHECK(
        grad_output.is_cuda() && grad_output.is_contiguous() &&
            grad_output.scalar_type() == torch::kFloat32 &&
            grad_output.numel() == 1,
        "grad_output must be a contiguous CUDA FP32 scalar");
    TORCH_CHECK(
        valid_count.is_cuda() && valid_count.is_contiguous() &&
            valid_count.scalar_type() == torch::kFloat32 &&
            valid_count.numel() == 1,
        "valid_count must be a contiguous CUDA FP32 scalar");
    TORCH_CHECK(
        grad_logits.device() == hidden.device() &&
            hidden.device() == weight.device() &&
            weight.device() == grad_output.device() &&
            grad_output.device() == valid_count.device(),
        "gradient-tail device mismatch");
    TORCH_CHECK(
        grad_logits.size(0) == hidden.size(0) &&
            grad_logits.size(1) == weight.size(0) &&
            hidden.size(1) == weight.size(1),
        "gradient-tail shape mismatch");

    const int M = static_cast<int>(hidden.size(0));
    const int K = static_cast<int>(hidden.size(1));
    const int V = static_cast<int>(weight.size(0));
    auto d_hidden = torch::empty_like(hidden);
    auto d_weight = torch::empty_like(weight);
    auto coefficients = torch::empty(
        {2}, torch::dtype(torch::kFloat32).device(hidden.device()));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    make_backward_coefficients_kernel<<<1, 1, 0, stream>>>(
        grad_output.data_ptr<float>(),
        valid_count.data_ptr<float>(),
        coefficients.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto handle = at::cuda::getCurrentCUDABlasHandle();
    CublasPointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_DEVICE);
    const float* alpha = coefficients.data_ptr<float>();
    const float* beta = alpha + 1;

    // Row-major d_hidden = grad_logits @ weight.
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        K,
        M,
        V,
        alpha,
        weight.data_ptr(),
        CUDA_R_16BF,
        K,
        grad_logits.data_ptr(),
        CUDA_R_16BF,
        V,
        beta,
        d_hidden.data_ptr(),
        CUDA_R_16BF,
        K,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Row-major d_weight = grad_logits.T @ hidden.
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        K,
        V,
        M,
        alpha,
        hidden.data_ptr(),
        CUDA_R_16BF,
        K,
        grad_logits.data_ptr(),
        CUDA_R_16BF,
        V,
        beta,
        d_weight.data_ptr(),
        CUDA_R_16BF,
        K,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    return std::make_tuple(d_hidden, d_weight);
}

std::tuple<torch::Tensor, torch::Tensor> valid_mask_count_cuda(
    torch::Tensor targets,
    int64_t ignore_index) {
    TORCH_CHECK(
        targets.is_cuda() && targets.is_contiguous() &&
            targets.scalar_type() == torch::kInt64,
        "targets must be contiguous CUDA int64");
    auto valid = torch::empty(
        targets.sizes(), torch::dtype(torch::kBool).device(targets.device()));
    auto valid_count = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(targets.device()));
    constexpr int threads = 256;
    const int blocks = std::min<int64_t>(
        (targets.numel() + threads - 1) / threads,
        128);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    valid_mask_count_kernel<<<std::max(blocks, 1), threads, 0, stream>>>(
        targets.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        valid_count.data_ptr<float>(),
        targets.numel(),
        ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(valid, valid_count);
}

torch::Tensor backward_scale_cuda(
    torch::Tensor grad_output,
    torch::Tensor valid_count) {
    TORCH_CHECK(
        grad_output.is_cuda() && grad_output.is_contiguous() &&
            grad_output.scalar_type() == torch::kFloat32 &&
            grad_output.numel() == 1,
        "grad_output must be a contiguous CUDA FP32 scalar");
    TORCH_CHECK(
        valid_count.is_cuda() && valid_count.is_contiguous() &&
            valid_count.scalar_type() == torch::kFloat32 &&
            valid_count.numel() == 1,
        "valid_count must be a contiguous CUDA FP32 scalar");
    TORCH_CHECK(
        grad_output.device() == valid_count.device(),
        "backward-scale device mismatch");
    auto coefficients = torch::empty(
        {2}, torch::dtype(torch::kFloat32).device(grad_output.device()));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    make_backward_coefficients_kernel<<<1, 1, 0, stream>>>(
        grad_output.data_ptr<float>(),
        valid_count.data_ptr<float>(),
        coefficients.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return coefficients.narrow(0, 0, 1);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "replace_target_logits_bf16",
        &replace_target_logits_bf16,
        "Replace target logits with BF16 dot products");
    m.def("softmax_loss_probs", &softmax_loss_probs,
          "Compute mean CE loss and BF16 softmax probabilities from BF16 logits");
    m.def("softmax_loss_probs_target_split", &softmax_loss_probs_target_split,
          "Compute CE loss, BF16 non-target probabilities, and exact target probabilities");
    m.def(
        "softmax_loss_probs_target_top1_split",
        &softmax_loss_probs_target_top1_split,
        "Compute CE loss and split exact target/top-1 probabilities from the BF16 FP4 tail");
    m.def(
        "softmax_loss_probs_target_top2_split",
        &softmax_loss_probs_target_top2_split,
        "Compute CE loss and split exact target/top-2 probabilities from the BF16 FP4 tail");
    m.def(
        "softmax_loss_probs_target_top4_split",
        &softmax_loss_probs_target_top4_split,
        "Compute CE loss and split exact target/top-4 probabilities from the BF16 FP4 tail");
    m.def("softmax_loss_grad_probs", &softmax_loss_grad_probs,
          "Compute mean CE loss and BF16 softmax gradients P-onehot from BF16 logits");
    m.def("softmax_loss_grad_probs_all_valid_full_vocab", &softmax_loss_grad_probs_all_valid_full_vocab,
          "Compute mean CE loss and BF16 softmax gradients for all-valid full-vocab batches");
    m.def("softmax_loss_lse", &softmax_loss_lse,
          "Compute mean CE loss and row log-sum-exp from BF16 logits");
    m.def(
        "softmax_loss_lse_target_topk_split",
        &softmax_loss_lse_target_topk_split,
        "Compute CE/LSE, split target/top-k probabilities, and mask them in-place");
    m.def(
        "softmax_loss_lse_target_topk_split_exact_logits",
        &softmax_loss_lse_target_topk_split_exact_logits,
        "Compute CE/LSE with exact BF16 target/top-k logits and an FP4 logit tail");
    m.def(
        "softmax_loss_lse_target_topk_split_exact_logits_nvfp4_row",
        &softmax_loss_lse_target_topk_split_exact_logits_nvfp4_row,
        "Compute repaired CE/LSE while emitting the NVFP4 probability-row operand");
    m.def(
        "softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row",
        &softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row,
        "Compute repaired CE/LSE while emitting the MXFP4 probability-row operand");
    m.def(
        "softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered",
        &softmax_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered,
        "Compute repaired CE/LSE from centered E4M3 logits while emitting the MXFP4 probability-row operand");
    m.def(
        "softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row",
        &softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row,
        "Compute repaired CE/LSE while emitting the MXFP8 probability-row operand");
    m.def(
        "softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row_with_sr_state",
        &softmax_loss_lse_target_topk_split_exact_logits_mxfp8_row_with_sr_state,
        "Compute repaired CE/LSE and MXFP8 row with explicit persistent SR state",
        py::arg("logits"),
        py::arg("x"),
        py::arg("weight"),
        py::arg("targets"),
        py::arg("valid"),
        py::arg("vocab_size"),
        py::arg("topk_split"),
        py::arg("quant_max"),
        py::arg("exact_selected_logits"),
        py::arg("persistent_row_sr_state"));
    m.def("softmax_lse", &softmax_lse,
          "Compute row log-sum-exp from BF16 logits");
    m.def("softmax_lse_targets", &softmax_lse_targets,
          "Compute row log-sum-exp plus local target logits/indices from BF16 logits");
    m.def("softmax_grad_probs_from_lse", &softmax_grad_probs_from_lse,
          "Compute BF16 softmax gradients from BF16 logits and precomputed row LSE");
    m.def("softmax_tail_grad_probs_from_lse", &softmax_tail_grad_probs_from_lse,
          "Compute temperature-scaled BF16 tail gradients from a corrected LSE");
    m.def(
        "softmax_repaired_grad_probs_from_lse",
        &softmax_repaired_grad_probs_from_lse,
        "Materialize BF16 tail G and scatter exact target/top-k coefficients");
    m.def(
        "softmax_repaired_grad_probs_from_lse_inplace",
        &softmax_repaired_grad_probs_from_lse_inplace,
        "Overwrite BF16 logits with repaired G and scatter exact coefficients");
    m.def("bf16_logits_cuda", &bf16_logits_cuda,
          "Compute BF16 logits = hidden @ weight.T with native CUDA/cuBLAS");
    m.def("bf16_tail_grads_cuda", &bf16_tail_grads_cuda,
          "Compute BF16 CCE dHidden/dWeight with native CUDA/cuBLAS");
    m.def("valid_mask_count_cuda", &valid_mask_count_cuda,
          "Compute CCE valid mask and count with native CUDA");
    m.def("backward_scale_cuda", &backward_scale_cuda,
          "Compute grad_output / valid_count with native CUDA");
}
