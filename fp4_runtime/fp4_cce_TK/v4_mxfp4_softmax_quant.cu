#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <tuple>

#define TK_STANDALONE
#include "../TK_quantisation/nvfp4_v5/core.cuh"

namespace {

constexpr int TILE = 128;
constexpr int FP4_BLOCK = 32;
constexpr int ROW_THREADS = 256;
constexpr uint8_t MXFP4_P_CONSTANT_E8M0 = 0x7f;
constexpr float MXFP4_P_CONSTANT_COEFF = 6.0f;
constexpr float LOG2_E = 1.4426950408889634f;
constexpr float LOG2_6 = 2.584962500721156f;
constexpr float LOG2_FLOAT_SCALE_ZERO = -126.23326760571977f;
constexpr float LOG_CODE_FAST_A = 1.50f;
constexpr float LOG_CODE_FAST_B = 1.20f;
constexpr float LOG_CODE_ACTIVATION_A = 1.60f;
constexpr float LOG_CODE_ACTIVATION_B = 0.95f;

using Mxfp4RNGState =
    transformer_engine::curanddx::detail::philox4x32_native_state<10>;

constexpr unsigned long long MXFP4_SR_INVOCATION_STRIDE = 1ull << 40;
constexpr unsigned long long MXFP4_SR_RANK_STRIDE = 1ull << 56;
__device__ unsigned long long mxfp4_sr_invocation_offset = 0;

__global__ void prepare_mxfp4_advancing_rng_state_kernel(
    unsigned long long* rng_state,
    unsigned long long rng_seed,
    unsigned long long rng_subsequence_base) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long offset = atomicAdd(
            &mxfp4_sr_invocation_offset, MXFP4_SR_INVOCATION_STRIDE);
        rng_state[0] = rng_seed;
        rng_state[1] = rng_subsequence_base + offset;
    }
}

torch::Tensor make_mxfp4_advancing_rng_state(
    const torch::Tensor& input,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream) {
    auto rng_state = torch::empty(
        {2}, torch::dtype(torch::kInt64).device(input.device()));
    prepare_mxfp4_advancing_rng_state_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
        static_cast<unsigned long long>(rng_seed),
        static_cast<unsigned long long>(rng_subsequence_base));
    return rng_state;
}

inline bool mxfp4_env_flag(const char* name, bool default_value = false) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return !(
        value[0] == '\0' || value[0] == '0' ||
        value[0] == 'f' || value[0] == 'F' ||
        value[0] == 'n' || value[0] == 'N');
}

inline uint64_t mxfp4_env_u64(const char* name, uint64_t default_value = 0) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
}

inline float mxfp8_logit_temperature() {
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

__device__ __forceinline__ uint32_t mxfp4_next_rbits(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ uint32_t mxfp4_stateless_seed(
    uint64_t seed,
    uint64_t sequence) {
    uint64_t value = seed ^ (sequence + 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    uint32_t result = static_cast<uint32_t>(value) ^
        static_cast<uint32_t>(value >> 32);
    return result == 0 ? 0x9e3779b9u : result;
}

__device__ __forceinline__ uint16_t select_stochastic_rounded_zeros(
    uint16_t deterministic,
    uint16_t stochastic) {
    const uint16_t magnitude = deterministic & 0x7777u;
    const uint16_t nonzero_lsb = static_cast<uint16_t>(
        (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x1111u);
    const uint16_t zero_mask = static_cast<uint16_t>(
        ((~nonzero_lsb) & 0x1111u) * 0x000fu);
    return static_cast<uint16_t>(
        (deterministic & ~zero_mask) | (stochastic & zero_mask));
}

__device__ __forceinline__ uint16_t round_positive_zeros_stochastic(
    const float2 in01,
    const float2 in23,
    float coefficient,
    uint32_t random_bits) {
    const float2 scaled01 = {
        fmaxf(in01.x * coefficient, 0.0f),
        fmaxf(in01.y * coefficient, 0.0f),
    };
    const float2 scaled23 = {
        fmaxf(in23.x * coefficient, 0.0f),
        fmaxf(in23.y * coefficient, 0.0f),
    };
    const uint8_t packed01 = static_cast<uint8_t>(
        __nv_cvt_float2_to_fp4x2(
            scaled01, __NV_E2M1, cudaRoundNearest));
    const uint8_t packed23 = static_cast<uint8_t>(
        __nv_cvt_float2_to_fp4x2(
            scaled23, __NV_E2M1, cudaRoundNearest));
    uint16_t packed = static_cast<uint16_t>(
        packed01 | (static_cast<uint16_t>(packed23) << 8));
    const float values[4] = {
        scaled01.x, scaled01.y, scaled23.x, scaled23.y};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const uint16_t nibble_mask = static_cast<uint16_t>(0x0fu << (lane * 4));
        if ((packed & nibble_mask) == 0 && values[lane] > 0.0f) {
            // Positive E2M1 has no subnormals: stochastic rounding below the
            // 0.5 minimum emits 0.5 with probability value / 0.5.
            const uint32_t threshold = static_cast<uint32_t>(fminf(
                values[lane] * 512.0f, 256.0f));
            const uint32_t random_byte =
                (random_bits >> (lane * 8)) & 0xffu;
            if (random_byte < threshold) {
                packed |= static_cast<uint16_t>(1u << (lane * 4));
            }
        }
    }
    return packed;
}

__device__ __forceinline__ void mul_cvt_fp32_to_fp4_4x_rn_and_rs(
    const float2 in01,
    const float2 in23,
    const float2 scale,
    uint32_t rbits,
    uint16_t& deterministic,
    uint16_t& stochastic) {
    uint32_t deterministic_out = 0;
    uint16_t stochastic_out = 0;
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        ".reg.b8 f0; \n\t"
        ".reg.b8 f1; \n\t"
        "mov.b64 {v0, v1}, %2; \n\t"
        "mov.b64 {v2, v3}, %3; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %4; \n\t"
        "mul.f32x2 v23, v23, %4; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1; \n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3; \n\t"
        "mov.b32 %0, {f0, f1, f0, f1}; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %1, {v2, v3, v0, v1}, %5; \n\t"
        "}"
        : "=r"(deterministic_out), "=h"(stochastic_out)
        : "l"(reinterpret_cast<const uint64_t&>(in01)),
          "l"(reinterpret_cast<const uint64_t&>(in23)),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(rbits));
    deterministic = static_cast<uint16_t>(deterministic_out);
    stochastic = stochastic_out;
}

__device__ __forceinline__ uint8_t float_to_e8m0_ceil(float val) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    if (mant > 0 && exp < 0xFE) ++exp;
    return exp;
}

__device__ __forceinline__ uint8_t float_to_e8m0_guarded_floor(
    float val,
    float floor_ratio) {
    if (val <= 1e-38f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    const uint32_t mant = u & 0x7FFFFF;
    if (mant == 0 || exp >= 0xFE) return exp;

    // E8M0 can only encode powers of two. Retain the standard ceiling unless
    // the value is close enough to the lower power that accepting limited
    // E2M1 saturation buys useful low-tail resolution.
    if (floor_ratio > 1.0f && exp > 0) {
        const float floor_value = __uint_as_float(
            static_cast<uint32_t>(exp) << 23);
        if (val <= floor_value * floor_ratio) return exp;
    }
    return exp + 1;
}

__device__ __forceinline__ float e8m0_decode(uint8_t e8m0) {
    return e8m0 == 0
        ? 0.0f
        : __uint_as_float(static_cast<uint32_t>(e8m0) << 23);
}

__device__ __forceinline__ float e8m0_quant_coeff(uint8_t e8m0) {
    if (e8m0 == 0) return 0.0f;
    if (e8m0 == 1) return __uint_as_float(0x7f800000u);
    // 6 * 2^(127-e) = 1.5 * 2^(129-e). Construct the exact normal
    // coefficient directly instead of issuing an SFU exp2. Code 1 would
    // overflow either formulation.
    return __uint_as_float(
        (static_cast<uint32_t>(256 - e8m0) << 23) | 0x00400000u);
}

__device__ __forceinline__ uint8_t log2_probability_to_e8m0(
    float block_max_log2_probability,
    float floor_log2_ratio) {
    if (!isfinite(block_max_log2_probability)) return 0;

    // The E8M0 value is the decoded block maximum. E2M1 payloads are
    // interpreted as q / 6, so the payload conversion (not the scale) carries
    // the factor of six.
    const float log2_scale = block_max_log2_probability;
    if (log2_scale <= LOG2_FLOAT_SCALE_ZERO) return 0;

    const float floor_exponent = floorf(log2_scale);
    const float fraction = log2_scale - floor_exponent;
    float selected_exponent = fraction == 0.0f
        ? floor_exponent
        : floor_exponent + 1.0f;
    if (floor_log2_ratio > 0.0f && fraction <= floor_log2_ratio) {
        selected_exponent = floor_exponent;
    }
    int code = static_cast<int>(selected_exponent) + 127;
    code = code < 1 ? 1 : (code > 254 ? 254 : code);
    return static_cast<uint8_t>(code);
}

__device__ __forceinline__ uint8_t positive_e2m1_code_from_log2(
    float log2_scaled_value) {
    if (!isfinite(log2_scaled_value) || log2_scaled_value < -2.0f) return 0;
    if (log2_scaled_value < -0.4150374992788438f) return 1;
    if (log2_scaled_value < 0.32192809488736235f) return 2;
    if (log2_scaled_value < 0.8073549220576041f) return 3;
    if (log2_scaled_value < 1.3219280948873624f) return 4;
    if (log2_scaled_value < 1.8073549220576042f) return 5;
    if (log2_scaled_value < 2.321928094887362f) return 6;
    return 7;
}

template <int LOG_CODE_MODE>
__device__ __forceinline__ uint8_t pack_positive_e2m1_pair_from_log2(
    float x0,
    float x1) {
    static_assert(
        LOG_CODE_MODE >= 1 && LOG_CODE_MODE <= 3,
        "LOG_CODE_MODE must select exact, fast-affine, or activation-affine");
    if constexpr (LOG_CODE_MODE == 1) {
        const uint8_t q0 = positive_e2m1_code_from_log2(x0);
        const uint8_t q1 = positive_e2m1_code_from_log2(x1);
        return static_cast<uint8_t>(q0 | (q1 << 4));
    } else {
        constexpr float A = LOG_CODE_MODE == 2
            ? LOG_CODE_FAST_A
            : LOG_CODE_ACTIVATION_A;
        constexpr float B = LOG_CODE_MODE == 2
            ? LOG_CODE_FAST_B
            : LOG_CODE_ACTIVATION_B;
        const float2 affine = {
            fmaxf(0.0f, fmaf(A, x0, B)),
            fmaxf(0.0f, fmaf(A, x1, B)),
        };
        return static_cast<uint8_t>(
            __nv_cvt_float2_to_fp4x2(
                affine, __NV_E2M1, cudaRoundNearest));
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

inline int select_staged_row_threads(const char* env_value) {
    if (env_value != nullptr) {
        if (env_value[0] == '1') return 1024;
        if (env_value[0] == '5') return 512;
        if (env_value[0] == '2') return 256;
    }
    return ROW_THREADS;
}

__device__ __forceinline__ void store_mxfp4_scales(
    uint8_t* __restrict__ scales,
    int tile_row,
    int tile_col,
    int ntk,
    int row_in_tile,
    const uint8_t vals[4]) {
    const int j = row_in_tile % 32;
    const int grp = row_in_tile / 32;
    const int base = (tile_row * ntk + tile_col) * 512 + j * 16 + grp * 4;
    const uint32_t packed =
        static_cast<uint32_t>(vals[0]) |
        (static_cast<uint32_t>(vals[1]) << 8) |
        (static_cast<uint32_t>(vals[2]) << 16) |
        (static_cast<uint32_t>(vals[3]) << 24);
    *reinterpret_cast<uint32_t*>(scales + base) = packed;
}

__device__ __forceinline__ void store_16_packed_fp4(uint8_t* __restrict__ dst, const uint8_t vals[16]) {
    uint4 packed;
    packed.x =
        static_cast<uint32_t>(vals[0]) |
        (static_cast<uint32_t>(vals[1]) << 8) |
        (static_cast<uint32_t>(vals[2]) << 16) |
        (static_cast<uint32_t>(vals[3]) << 24);
    packed.y =
        static_cast<uint32_t>(vals[4]) |
        (static_cast<uint32_t>(vals[5]) << 8) |
        (static_cast<uint32_t>(vals[6]) << 16) |
        (static_cast<uint32_t>(vals[7]) << 24);
    packed.z =
        static_cast<uint32_t>(vals[8]) |
        (static_cast<uint32_t>(vals[9]) << 8) |
        (static_cast<uint32_t>(vals[10]) << 16) |
        (static_cast<uint32_t>(vals[11]) << 24);
    packed.w =
        static_cast<uint32_t>(vals[12]) |
        (static_cast<uint32_t>(vals[13]) << 8) |
        (static_cast<uint32_t>(vals[14]) << 16) |
        (static_cast<uint32_t>(vals[15]) << 24);
    *reinterpret_cast<uint4*>(dst) = packed;
}

template <int LOG_CODE_MODE>
__global__ __launch_bounds__(256)
void mxfp4_softmax_log_quant_row_col_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const bool* __restrict__ valid,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float floor_log2_ratio) {
    // Keep logits rather than probabilities in shared memory. Once LSE is
    // known, both E8M0 scale selection and E2M1 rounding decisions can be
    // made in log2 space without evaluating exp for every tail value.
    __shared__ __nv_bfloat16 score_tile[TILE][TILE + 1];

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
        float z = -INFINITY;
        if (gr < M && gc < vocab_size && valid[gr]) {
            z = __bfloat162float(logits[static_cast<int64_t>(gr) * N + gc]);
        }
        score_tile[r][c] = __float2bfloat16(z);
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        const float row_lse = lse[gr];
        uint8_t scales[4];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float block_max_log2_probability = -INFINITY;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float z = __bfloat162float(
                    score_tile[r][b * FP4_BLOCK + j]);
                block_max_log2_probability = fmaxf(
                    block_max_log2_probability,
                    (z - row_lse) * LOG2_E);
            }
            const uint8_t e8m0 = log2_probability_to_e8m0(
                block_max_log2_probability, floor_log2_ratio);
            scales[b] = e8m0;
            const float scale_exponent = static_cast<float>(e8m0) - 127.0f;

            uint8_t packed_vals[16];
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; j += 2) {
                const int c = b * FP4_BLOCK + j;
                const float z0 = __bfloat162float(score_tile[r][c]);
                const float z1 = __bfloat162float(score_tile[r][c + 1]);
                packed_vals[j >> 1] = e8m0 == 0
                    ? 0
                    : pack_positive_e2m1_pair_from_log2<LOG_CODE_MODE>(
                        (z0 - row_lse) * LOG2_E - scale_exponent + LOG2_6,
                        (z1 - row_lse) * LOG2_E - scale_exponent + LOG2_6);
            }
            store_16_packed_fp4(
                reinterpret_cast<uint8_t*>(row_fp4) +
                    static_cast<int64_t>(gr) * (N / 2) +
                    (col_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_mxfp4_scales(
            row_sc, tile_row, tile_col, N / TILE, r, scales);
    } else if (tid < 2 * TILE) {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scales[4];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float block_max_log2_probability = -INFINITY;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const int r = b * FP4_BLOCK + j;
                const int gr = row_base + r;
                const float z = __bfloat162float(score_tile[r][c]);
                block_max_log2_probability = fmaxf(
                    block_max_log2_probability,
                    (z - lse[gr]) * LOG2_E);
            }
            const uint8_t e8m0 = log2_probability_to_e8m0(
                block_max_log2_probability, floor_log2_ratio);
            scales[b] = e8m0;
            const float scale_exponent = static_cast<float>(e8m0) - 127.0f;

            uint8_t packed_vals[16];
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; j += 2) {
                const int r = b * FP4_BLOCK + j;
                const int gr0 = row_base + r;
                const int gr1 = gr0 + 1;
                const float z0 = __bfloat162float(score_tile[r][c]);
                const float z1 = __bfloat162float(score_tile[r + 1][c]);
                packed_vals[j >> 1] = e8m0 == 0
                    ? 0
                    : pack_positive_e2m1_pair_from_log2<LOG_CODE_MODE>(
                        (z0 - lse[gr0]) * LOG2_E - scale_exponent + LOG2_6,
                        (z1 - lse[gr1]) * LOG2_E - scale_exponent + LOG2_6);
            }
            store_16_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(gc) * (M / 2) +
                    (row_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_mxfp4_scales(
            col_sc, tile_col, tile_row, M / TILE, c, scales);
    }
}

template <
    bool CONSTANT_SCALE,
    bool GRAD_CACHE,
    bool ROW_DATA_SR = false,
    bool COL_DATA_SR = false,
    bool COL_ZERO_SR = false>
__global__ __launch_bounds__(256)
void mxfp4_softmax_quant_row_col_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ lse,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    int vocab_size,
    float logit_temperature,
    float scale_floor_ratio,
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

    if constexpr (ROW_DATA_SR || COL_DATA_SR || COL_ZERO_SR) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
    }
    Mxfp4RNGState rng;
    uint32_t sr_state = 0;
    if constexpr (ROW_DATA_SR || COL_DATA_SR || COL_ZERO_SR) {
        const uint64_t tile =
            static_cast<uint64_t>(tile_row) * gridDim.x + tile_col;
        rng.init(
            rng_seed,
            rng_subsequence_base + tile * 256 + tid,
            0);
        const uint4 initial_random = rng.generate4();
        sr_state = initial_random.x ^ initial_random.y ^
            initial_random.z ^ initial_random.w;
        if (sr_state == 0) sr_state = 0x9e3779b9u;
    }

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
            if constexpr (GRAD_CACHE) {
                if (targets != nullptr && gc == targets[gr]) {
                    p -= 1.0f;
                }
            }
        }
        probs[r][c] = __float2bfloat16(p);
    }
    __syncthreads();

    if (tid < TILE) {
        const int r = tid;
        const int gr = row_base + r;
        uint8_t scales[4];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float amax = 0.0f;
            if constexpr (!CONSTANT_SCALE) {
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; ++j) {
                    const float p = __bfloat162float(probs[r][b * FP4_BLOCK + j]);
                    amax = fmaxf(amax, GRAD_CACHE ? fabsf(p) : p);
                }
            }
            const uint8_t e8m0 = CONSTANT_SCALE
                ? MXFP4_P_CONSTANT_E8M0
                : (ROW_DATA_SR
                    ? float_to_e8m0_ceil(amax)
                    : float_to_e8m0_guarded_floor(amax, scale_floor_ratio));
            scales[b] = e8m0;
            const float coeff = CONSTANT_SCALE ? MXFP4_P_CONSTANT_COEFF : e8m0_quant_coeff(e8m0);

            uint8_t packed_vals[16];
            if constexpr (ROW_DATA_SR) {
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; j += 4) {
                    const int c = b * FP4_BLOCK + j;
                    const float2 in01 = {
                        __bfloat162float(probs[r][c]),
                        __bfloat162float(probs[r][c + 1]),
                    };
                    const float2 in23 = {
                        __bfloat162float(probs[r][c + 2]),
                        __bfloat162float(probs[r][c + 3]),
                    };
                    const float2 scale = {coeff, coeff};
                    const auto packed4 =
                        transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                            in01,
                            in23,
                            scale,
                            mxfp4_next_rbits(sr_state));
                    const uint16_t raw =
                        *reinterpret_cast<const uint16_t*>(&packed4);
                    packed_vals[j >> 1] = static_cast<uint8_t>(raw & 0xffu);
                    packed_vals[(j >> 1) + 1] =
                        static_cast<uint8_t>((raw >> 8) & 0xffu);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; j += 4) {
                    const int c = b * FP4_BLOCK + j;
                    const float2 in01 = {
                        __bfloat162float(probs[r][c]),
                        __bfloat162float(probs[r][c + 1]),
                    };
                    const float2 in23 = {
                        __bfloat162float(probs[r][c + 2]),
                        __bfloat162float(probs[r][c + 3]),
                    };
                    const float2 scale = {coeff, coeff};
                    const auto packed4 =
                        transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<false>(
                            in01, in23, scale, 0);
                    const uint16_t raw =
                        *reinterpret_cast<const uint16_t*>(&packed4);
                    packed_vals[j >> 1] = static_cast<uint8_t>(raw & 0xffu);
                    packed_vals[(j >> 1) + 1] =
                        static_cast<uint8_t>((raw >> 8) & 0xffu);
                }
            }
            store_16_packed_fp4(
                reinterpret_cast<uint8_t*>(row_fp4) + static_cast<int64_t>(gr) * (N / 2) + (col_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_mxfp4_scales(row_sc, tile_row, tile_col, N / TILE, r, scales);
    } else if (tid < 2 * TILE) {
        const int c = tid - TILE;
        const int gc = col_base + c;
        uint8_t scales[4];

        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            float amax = 0.0f;
            if constexpr (!CONSTANT_SCALE) {
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; ++j) {
                    const float p = __bfloat162float(probs[b * FP4_BLOCK + j][c]);
                    amax = fmaxf(amax, GRAD_CACHE ? fabsf(p) : p);
                }
            }
            const uint8_t e8m0 = CONSTANT_SCALE
                ? MXFP4_P_CONSTANT_E8M0
                : (COL_DATA_SR
                    ? float_to_e8m0_ceil(amax)
                    : float_to_e8m0_guarded_floor(amax, scale_floor_ratio));
            scales[b] = e8m0;
            const float coeff = CONSTANT_SCALE ? MXFP4_P_CONSTANT_COEFF : e8m0_quant_coeff(e8m0);

            uint8_t packed_vals[16];
            if constexpr (COL_DATA_SR) {
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; j += 4) {
                    const int r = b * FP4_BLOCK + j;
                    const float2 in01 = {
                        __bfloat162float(probs[r][c]),
                        __bfloat162float(probs[r + 1][c]),
                    };
                    const float2 in23 = {
                        __bfloat162float(probs[r + 2][c]),
                        __bfloat162float(probs[r + 3][c]),
                    };
                    const float2 scale = {coeff, coeff};
                    const auto packed4 =
                        transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                            in01,
                            in23,
                            scale,
                            mxfp4_next_rbits(sr_state));
                    const uint16_t raw =
                        *reinterpret_cast<const uint16_t*>(&packed4);
                    packed_vals[j >> 1] = static_cast<uint8_t>(raw & 0xffu);
                    packed_vals[(j >> 1) + 1] =
                        static_cast<uint8_t>((raw >> 8) & 0xffu);
                }
            } else {
                #pragma unroll
                for (int j = 0; j < FP4_BLOCK; j += 4) {
                    const int r = b * FP4_BLOCK + j;
                    const float2 in01 = {
                        __bfloat162float(probs[r][c]),
                        __bfloat162float(probs[r + 1][c]),
                    };
                    const float2 in23 = {
                        __bfloat162float(probs[r + 2][c]),
                        __bfloat162float(probs[r + 3][c]),
                    };
                    const float2 scale = {coeff, coeff};
                    if constexpr (COL_ZERO_SR) {
                        uint16_t raw;
                        uint16_t stochastic_raw;
                        mul_cvt_fp32_to_fp4_4x_rn_and_rs(
                            in01,
                            in23,
                            scale,
                            mxfp4_next_rbits(sr_state),
                            raw,
                            stochastic_raw);
                        raw = select_stochastic_rounded_zeros(
                            raw, stochastic_raw);
                        packed_vals[j >> 1] =
                            static_cast<uint8_t>(raw & 0xffu);
                        packed_vals[(j >> 1) + 1] =
                            static_cast<uint8_t>((raw >> 8) & 0xffu);
                    } else {
                        const auto packed4 =
                            transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<false>(
                                in01, in23, scale, 0);
                        const uint16_t raw =
                            *reinterpret_cast<const uint16_t*>(&packed4);
                        packed_vals[j >> 1] =
                            static_cast<uint8_t>(raw & 0xffu);
                        packed_vals[(j >> 1) + 1] =
                            static_cast<uint8_t>((raw >> 8) & 0xffu);
                    }
                }
            }
            store_16_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) + static_cast<int64_t>(gc) * (M / 2) + (row_base + b * FP4_BLOCK) / 2,
                packed_vals);
        }
        store_mxfp4_scales(col_sc, tile_col, tile_row, M / TILE, c, scales);
    }
}

template <
    int ROW_TILES,
    int ROWS_PER_OUTPUT_TILE = TILE,
    bool APPLY_ROW_NORMALIZATION,
    bool COL_ZERO_SR = false,
    bool COL_FULL_SR = false,
    bool COL_DIRECT_ZERO_SR = false>
__global__ __launch_bounds__(ROW_TILES * TILE)
void mxfp4_col_requant_from_row_kernel(
    const __nv_fp4x2_e2m1* __restrict__ row_fp4,
    const uint8_t* __restrict__ row_sc,
    const float* __restrict__ row_normalization,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N,
    float scale_floor_ratio,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    const uint64_t* __restrict__ rng_state) {
    // Two columns keep each row four-byte aligned for packed BF16 stores while
    // still breaking the transpose bank-stride pattern.
    static_assert(
        ROW_TILES == 1 || ROW_TILES == 2,
        "column requantization supports one or two output slices per CTA");
    static_assert(
        ROWS_PER_OUTPUT_TILE == 32 || ROWS_PER_OUTPUT_TILE == 64 ||
            ROWS_PER_OUTPUT_TILE == TILE,
        "column requantization slice must contain 32, 64, or 128 rows");
    static_assert(
        ROWS_PER_OUTPUT_TILE == TILE || ROW_TILES == 1,
        "sub-128 row slices use one output slice per CTA");
    static_assert(
        !(COL_ZERO_SR && COL_FULL_SR),
        "zero-only and full column stochastic rounding are mutually exclusive");
    static_assert(
        !COL_DIRECT_ZERO_SR || COL_ZERO_SR,
        "direct zero rescue requires zero-only column stochastic rounding");
    constexpr int ROWS_PER_CTA = ROW_TILES * ROWS_PER_OUTPUT_TILE;
    constexpr int BLOCKS_PER_OUTPUT = ROWS_PER_OUTPUT_TILE / FP4_BLOCK;
    __shared__ __nv_bfloat16 values[ROWS_PER_CTA][TILE + 2];
    const int tile_col = blockIdx.x;
    const int tile_row_group = blockIdx.y;
    const int row_base = tile_row_group * ROWS_PER_CTA;
    const int col_base = tile_col * TILE;
    const int tid = threadIdx.x;
    const int tiles_per_row = N / TILE;
    if constexpr (COL_ZERO_SR || COL_FULL_SR) {
        if (rng_state != nullptr) {
            rng_seed = rng_state[0];
            rng_subsequence_base = rng_state[1];
        }
    }
    uint32_t sr_state = 0;
    if constexpr (COL_ZERO_SR || COL_FULL_SR) {
        const uint64_t tile =
            static_cast<uint64_t>(tile_row_group) * gridDim.x + tile_col;
        sr_state = mxfp4_stateless_seed(
            rng_seed,
            rng_subsequence_base + tile * 256 + tid);
    }

    for (int block = tid;
         block < ROWS_PER_CTA * (TILE / FP4_BLOCK);
         block += blockDim.x) {
        const int r = block / (TILE / FP4_BLOCK);
        const int b = block & ((TILE / FP4_BLOCK) - 1);
        const int gr = row_base + r;
        const int source_tile_row =
            gr / TILE;
        const int row_in_tile = gr % TILE;
        const int scale_offset =
            (source_tile_row * tiles_per_row + tile_col) * 512 +
            (row_in_tile & 31) * 16 + (row_in_tile >> 5) * 4 + b;
        const uint8_t scale_raw = row_sc[scale_offset];
        float represented_scale = e8m0_decode(scale_raw) * (1.0f / 6.0f);
        if constexpr (APPLY_ROW_NORMALIZATION) {
            represented_scale *= row_normalization[gr];
        }
        const int64_t packed_base =
            static_cast<int64_t>(gr) * (N / 2) +
            (col_base + b * FP4_BLOCK) / 2;
        const uint4 packed = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(row_fp4) + packed_base);
        const auto* packed_bytes = reinterpret_cast<const uint8_t*>(&packed);
        #pragma unroll
        for (int pair = 0; pair < FP4_BLOCK / 2; ++pair) {
            __nv_fp4x2_e2m1 packed_pair;
            packed_pair.__x = static_cast<__nv_fp4x2_storage_t>(
                packed_bytes[pair]);
            const float2 q = static_cast<float2>(packed_pair);
            *reinterpret_cast<__nv_bfloat162*>(
                &values[r][b * FP4_BLOCK + pair * 2]) =
                __floats2bfloat162_rn(
                    q.x * represented_scale,
                    q.y * represented_scale);
        }
    }
    __syncthreads();

    {
        const int c = tid % TILE;
        const int row_tile = tid / TILE;
        const int output_row_base =
            row_base + row_tile * ROWS_PER_OUTPUT_TILE;
        const int gc = col_base + c;
        uint8_t scales[BLOCKS_PER_OUTPUT];
        #pragma unroll
        for (int b = 0; b < BLOCKS_PER_OUTPUT; ++b) {
            float block_values[FP4_BLOCK];
            float block_amax = 0.0f;
            #pragma unroll
            for (int j = 0; j < FP4_BLOCK; ++j) {
                const float value = __bfloat162float(
                    values[
                        row_tile * ROWS_PER_OUTPUT_TILE +
                        b * FP4_BLOCK + j][c]);
                block_values[j] = value;
                block_amax = fmaxf(block_amax, fabsf(value));
            }
            const uint8_t scale = float_to_e8m0_guarded_floor(
                block_amax, scale_floor_ratio);
            scales[b] = scale;
            const float coefficient = e8m0_quant_coeff(scale);
            uint8_t packed_values[FP4_BLOCK / 2];
            if constexpr (COL_FULL_SR) {
                #pragma unroll
                for (int element = 0; element < FP4_BLOCK; element += 4) {
                    const float2 in01 = {
                        block_values[element],
                        block_values[element + 1],
                    };
                    const float2 in23 = {
                        block_values[element + 2],
                        block_values[element + 3],
                    };
                    const float2 scale_pair = {coefficient, coefficient};
                    const auto packed4 =
                        transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                            in01,
                            in23,
                            scale_pair,
                            mxfp4_next_rbits(sr_state));
                    const uint16_t raw =
                        *reinterpret_cast<const uint16_t*>(&packed4);
                    packed_values[element / 2] =
                        static_cast<uint8_t>(raw & 0xffu);
                    packed_values[element / 2 + 1] =
                        static_cast<uint8_t>((raw >> 8) & 0xffu);
                }
            } else if constexpr (COL_ZERO_SR) {
                #pragma unroll
                for (int element = 0; element < FP4_BLOCK; element += 4) {
                    const float2 in01 = {
                        block_values[element],
                        block_values[element + 1],
                    };
                    const float2 in23 = {
                        block_values[element + 2],
                        block_values[element + 3],
                    };
                    uint16_t rescued;
                    if constexpr (COL_DIRECT_ZERO_SR) {
                        rescued = round_positive_zeros_stochastic(
                            in01,
                            in23,
                            coefficient,
                            mxfp4_next_rbits(sr_state));
                    } else {
                        const float2 scale_pair = {coefficient, coefficient};
                        uint16_t deterministic;
                        uint16_t stochastic;
                        mul_cvt_fp32_to_fp4_4x_rn_and_rs(
                            in01,
                            in23,
                            scale_pair,
                            mxfp4_next_rbits(sr_state),
                            deterministic,
                            stochastic);
                        rescued = select_stochastic_rounded_zeros(
                            deterministic, stochastic);
                    }
                    packed_values[element / 2] =
                        static_cast<uint8_t>(rescued & 0xffu);
                    packed_values[element / 2 + 1] =
                        static_cast<uint8_t>((rescued >> 8) & 0xffu);
                }
            } else {
                #pragma unroll
                for (int pair = 0; pair < FP4_BLOCK / 2; ++pair) {
                    const float2 scaled = {
                        block_values[pair * 2] * coefficient,
                        block_values[pair * 2 + 1] * coefficient,
                    };
                    packed_values[pair] = static_cast<uint8_t>(
                        __nv_cvt_float2_to_fp4x2(
                            scaled, __NV_E2M1, cudaRoundNearest));
                }
            }
            store_16_packed_fp4(
                reinterpret_cast<uint8_t*>(col_fp4) +
                    static_cast<int64_t>(gc) * (M / 2) +
                    (output_row_base + b * FP4_BLOCK) / 2,
                packed_values);
        }
        if constexpr (ROWS_PER_OUTPUT_TILE == TILE) {
            store_mxfp4_scales(
                col_sc,
                tile_col,
                output_row_base / TILE,
                M / TILE,
                c,
                scales);
        } else {
            const int output_tile_col = output_row_base / TILE;
            const int scale_offset = (output_row_base % TILE) / FP4_BLOCK;
            const int j = c % 32;
            const int grp = c / 32;
            const int base =
                (tile_col * (M / TILE) + output_tile_col) * 512 +
                j * 16 + grp * 4 + scale_offset;
            if constexpr (BLOCKS_PER_OUTPUT == 1) {
                col_sc[base] = scales[0];
            } else {
                const uint16_t packed =
                    static_cast<uint16_t>(scales[0]) |
                    (static_cast<uint16_t>(scales[1]) << 8);
                *reinterpret_cast<uint16_t*>(col_sc + base) = packed;
            }
        }
    }
}

template <int THREADS, bool STAGE_EXP, bool CONSTANT_SCALE>
__global__ __launch_bounds__(THREADS)
void mxfp4_softmax_row_quant_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const bool* __restrict__ valid,
    __nv_bfloat16* __restrict__ probs,
    __nv_fp4x2_e2m1* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    float* __restrict__ loss_sum,
    float* __restrict__ count_sum,
    int M,
    int N,
    int vocab_size,
    bool grad_cache) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    constexpr int THREAD_WARPS = THREADS / 32;
    __shared__ float warp_red[THREAD_WARPS];
    __shared__ float s_max;
    __shared__ float s_inv_sum;
    __shared__ float s_lse;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const bool is_valid = valid[row];
    const int64_t target = is_valid ? targets[row] : -1;
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
        float block_max = (lane < THREAD_WARPS) ? warp_red[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max = block_max;
    }
    __syncthreads();

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
        float sum = (lane < THREAD_WARPS) ? warp_red[lane] : 0.0f;
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
    const int ntk = N / TILE;
    const int j = row_in_tile % 32;
    const int grp = row_in_tile / 32;

    for (int block = tid; block < blocks; block += THREADS) {
        const int col0 = block * FP4_BLOCK;
        const int tile_col = col0 / TILE;
        const int scale_idx = (col0 / FP4_BLOCK) & 3;
        float vals[FP4_BLOCK];
        float amax = 0.0f;

        #pragma unroll
        for (int k = 0; k < FP4_BLOCK; ++k) {
            const int col = col0 + k;
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
            vals[k] = p;
            if constexpr (!CONSTANT_SCALE) {
                amax = fmaxf(amax, fabsf(p));
            }
            probs[static_cast<int64_t>(row) * N + col] = __float2bfloat16(p);
        }

        const uint8_t e8m0 = CONSTANT_SCALE ? MXFP4_P_CONSTANT_E8M0 : float_to_e8m0_ceil(amax);
        const float coeff = CONSTANT_SCALE ? MXFP4_P_CONSTANT_COEFF : e8m0_quant_coeff(e8m0);
        const int scale_base = (tile_row * ntk + tile_col) * 512 + j * 16 + grp * 4 + scale_idx;
        row_sc[scale_base] = e8m0;

        uint8_t packed_vals[16];
        #pragma unroll
        for (int k = 0; k < FP4_BLOCK; k += 2) {
            float2 scaled = {vals[k] * coeff, vals[k + 1] * coeff};
            packed_vals[k >> 1] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
        }
    store_16_packed_fp4(
        reinterpret_cast<uint8_t*>(row_fp4) + static_cast<int64_t>(row) * (N / 2) + col0 / 2,
        packed_vals);
    }
}

template <int THREADS>
void launch_mxfp4_softmax_row_quant_staged(
    const __nv_bfloat16* logits,
    const int64_t* targets,
    const bool* valid,
    __nv_bfloat16* probs,
    __nv_fp4x2_e2m1* row_fp4,
    uint8_t* row_sc,
    float* loss_sum,
    float* count_sum,
    int M,
    int N,
    int vocab_size,
    bool stage_exp,
    bool constant_scale,
    bool grad_cache,
    cudaStream_t stream) {
    if (stage_exp && constant_scale) {
        mxfp4_softmax_row_quant_kernel<THREADS, true, true><<<M, THREADS, 0, stream>>>(
            logits, targets, valid, probs, row_fp4, row_sc, loss_sum, count_sum, M, N, vocab_size, grad_cache);
    } else if (stage_exp) {
        mxfp4_softmax_row_quant_kernel<THREADS, true, false><<<M, THREADS, 0, stream>>>(
            logits, targets, valid, probs, row_fp4, row_sc, loss_sum, count_sum, M, N, vocab_size, grad_cache);
    } else if (constant_scale) {
        mxfp4_softmax_row_quant_kernel<THREADS, false, true><<<M, THREADS, 0, stream>>>(
            logits, targets, valid, probs, row_fp4, row_sc, loss_sum, count_sum, M, N, vocab_size, grad_cache);
    } else {
        mxfp4_softmax_row_quant_kernel<THREADS, false, false><<<M, THREADS, 0, stream>>>(
            logits, targets, valid, probs, row_fp4, row_sc, loss_sum, count_sum, M, N, vocab_size, grad_cache);
    }
}

template <bool CONSTANT_SCALE>
__global__ __launch_bounds__(128)
void mxfp4_col_quant_from_probs_kernel(
    const __nv_bfloat16* __restrict__ probs,
    __nv_fp4x2_e2m1* __restrict__ col_fp4,
    uint8_t* __restrict__ col_sc,
    int M,
    int N) {
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int row_base = tile_row * TILE;
    const int col_base = tile_col * TILE;
    const int c = threadIdx.x;
    if (c >= TILE) return;

    const int gc = col_base + c;
    const int ntk = M / TILE;
    uint8_t scales[4];

    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        float vals[FP4_BLOCK];
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < FP4_BLOCK; ++j) {
            const int r = b * FP4_BLOCK + j;
            const float p = __bfloat162float(probs[static_cast<int64_t>(row_base + r) * N + gc]);
            vals[j] = p;
            if constexpr (!CONSTANT_SCALE) {
                amax = fmaxf(amax, fabsf(p));
            }
        }
        const uint8_t e8m0 = CONSTANT_SCALE ? MXFP4_P_CONSTANT_E8M0 : float_to_e8m0_ceil(amax);
        scales[b] = e8m0;
        const float coeff = CONSTANT_SCALE ? MXFP4_P_CONSTANT_COEFF : e8m0_quant_coeff(e8m0);

        uint8_t packed_vals[16];
        #pragma unroll
        for (int j = 0; j < FP4_BLOCK; j += 2) {
            float2 scaled = {vals[j] * coeff, vals[j + 1] * coeff};
            packed_vals[j >> 1] = static_cast<uint8_t>(
                __nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
        }
        store_16_packed_fp4(
            reinterpret_cast<uint8_t*>(col_fp4) + static_cast<int64_t>(gc) * (M / 2) + (row_base + b * FP4_BLOCK) / 2,
            packed_vals);
    }
    store_mxfp4_scales(col_sc, tile_col, tile_row, ntk, c, scales);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_quant_row_col_with_floor_ratio(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size,
    double scale_floor_ratio) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat, "lse must be contiguous CUDA float");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool, "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(lse.numel() == M64 && valid.numel() == M64, "lse/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");
    TORCH_CHECK(
        scale_floor_ratio >= 1.0 && scale_floor_ratio <= 2.0,
        "scale_floor_ratio must be in [1, 2]");

    auto device = logits.device();
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));

    dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const char* use_log_code = std::getenv("FP4_CCE_V4_MXFP4_G_LOG_CODE");
    const int log_code_mode = use_log_code == nullptr
        ? 0
        : std::atoi(use_log_code);
    const float logit_temperature = mxfp8_logit_temperature();
    TORCH_CHECK(
        log_code_mode >= 0 && log_code_mode <= 3,
        "FP4_CCE_V4_MXFP4_G_LOG_CODE must be 0, 1, 2, or 3");
    TORCH_CHECK(
        log_code_mode == 0 || logit_temperature == 1.0f,
        "logit temperature is not supported with MXFP4 log-code modes");
    const bool data_sr = mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_DATA_SR");
    const bool row_data_sr = data_sr ||
        mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_ROW_DATA_SR");
    const bool col_data_sr = data_sr ||
        mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_COL_DATA_SR");
    const bool col_zero_sr =
        mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_COL_ZERO_SR");
    TORCH_CHECK(
        !col_zero_sr || !col_data_sr,
        "MXFP4 column full SR and zero-only SR are mutually exclusive");
    const bool any_data_sr = row_data_sr || col_data_sr || col_zero_sr;
    TORCH_CHECK(
        !any_data_sr || log_code_mode == 0,
        "MXFP4 log-code modes and MXFP4 data SR are separate experiments");
    const uint64_t rng_seed =
        mxfp4_env_u64("FP4_CCE_V4_MXFP4_G_RNG_SEED", 0);
    const uint64_t rng_subsequence_base =
        mxfp4_env_u64("FP4_CCE_V4_MXFP4_G_RNG_SUBSEQUENCE_BASE", 0) +
        mxfp4_env_u64("RANK", 0) * MXFP4_SR_RANK_STRIDE;
    auto rng_state = any_data_sr
        ? make_mxfp4_advancing_rng_state(
            logits, rng_seed, rng_subsequence_base, stream)
        : torch::Tensor();
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    const char* use_constant = std::getenv("FP4_CCE_V4_MXFP4_TILED_CONSTANT_SCALE");
#define LAUNCH_MXFP4_TAIL_QUANT(CONSTANT, GRAD, ROW_SR, COL_SR, COL_ZERO_SR) \
    mxfp4_softmax_quant_row_col_kernel< \
        CONSTANT, GRAD, ROW_SR, COL_SR, COL_ZERO_SR><<<grid, 256, 0, stream>>>( \
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()), \
        lse.data_ptr<float>(), \
        nullptr, \
        valid.data_ptr<bool>(), \
        reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()), \
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()), \
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()), \
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()), \
        static_cast<int>(M64), \
        static_cast<int>(N64), \
        static_cast<int>(vocab_size), \
        logit_temperature, \
        static_cast<float>(scale_floor_ratio), \
        rng_seed, \
        rng_subsequence_base, \
        rng_state_ptr)
    if (log_code_mode == 1) {
        mxfp4_softmax_log_quant_row_col_kernel<1><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            static_cast<float>(std::log2(scale_floor_ratio)));
    } else if (log_code_mode == 2) {
        mxfp4_softmax_log_quant_row_col_kernel<2><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            static_cast<float>(std::log2(scale_floor_ratio)));
    } else if (log_code_mode == 3) {
        mxfp4_softmax_log_quant_row_col_kernel<3><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            lse.data_ptr<float>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            static_cast<float>(std::log2(scale_floor_ratio)));
    } else if (use_constant != nullptr && use_constant[0] != '0') {
        if (row_data_sr && col_data_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(true, false, true, true, false);
        } else if (row_data_sr && col_zero_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(true, false, true, false, true);
        } else if (row_data_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(true, false, true, false, false);
        } else if (col_data_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(true, false, false, true, false);
        } else if (col_zero_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(true, false, false, false, true);
        } else {
            LAUNCH_MXFP4_TAIL_QUANT(true, false, false, false, false);
        }
    } else {
        // The GRAD_CACHE specialization generates materially faster code on
        // GB200. With a null target pointer it emits the same nonnegative
        // softmax tail, including bit-identical block scales and FP4 values.
        if (row_data_sr && col_data_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(false, true, true, true, false);
        } else if (row_data_sr && col_zero_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(false, true, true, false, true);
        } else if (row_data_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(false, true, true, false, false);
        } else if (col_data_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(false, true, false, true, false);
        } else if (col_zero_sr) {
            LAUNCH_MXFP4_TAIL_QUANT(false, true, false, false, true);
        } else {
            LAUNCH_MXFP4_TAIL_QUANT(false, true, false, false, false);
        }
    }
#undef LAUNCH_MXFP4_TAIL_QUANT
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_softmax_quant_row_col failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_quant_row_col(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor valid,
    int64_t vocab_size) {
    return mxfp4_softmax_quant_row_col_with_floor_ratio(
        logits, lse, valid, vocab_size, 1.0);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_col_requant_from_row(
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor row_normalization,
    double scale_floor_ratio) {
    TORCH_CHECK(
        row_fp4.is_cuda() && row_fp4.is_contiguous() && row_fp4.dim() == 2 &&
            row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2,
        "row_fp4 must be contiguous CUDA float4_e2m1fn_x2 [M, N/2]");
    TORCH_CHECK(
        row_sc.is_cuda() && row_sc.is_contiguous() && row_sc.dim() == 4 &&
            row_sc.scalar_type() == torch::kUInt8,
        "row_sc must be contiguous CUDA uint8 MXFP4 scales");
    TORCH_CHECK(
        row_normalization.is_cuda() && row_normalization.is_contiguous() &&
            row_normalization.scalar_type() == torch::kFloat32 &&
            row_normalization.dim() == 1,
        "row_normalization must be contiguous CUDA float32 [M]");
    TORCH_CHECK(
        row_fp4.device() == row_sc.device() &&
            row_fp4.device() == row_normalization.device(),
        "row values, scales, and normalization must be on one CUDA device");
    TORCH_CHECK(
        std::isfinite(scale_floor_ratio) && scale_floor_ratio >= 1.0 &&
            scale_floor_ratio <= 2.0,
        "scale_floor_ratio must be finite and in [1, 2]");

    const int64_t M64 = row_fp4.size(0);
    const int64_t N64 = row_fp4.size(1) * 2;
    TORCH_CHECK(
        M64 % TILE == 0 && N64 % TILE == 0,
        "row-requantized dimensions must be multiples of 128");
    TORCH_CHECK(
        row_sc.size(0) == M64 / TILE && row_sc.size(1) == N64 / TILE &&
            row_sc.size(2) == 32 && row_sc.size(3) == 16,
        "row scale shape mismatch");
    TORCH_CHECK(
        row_normalization.numel() == M64,
        "row normalization length mismatch");

    auto device = row_fp4.device();
    auto col_fp4 = torch::empty(
        {N64, M64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {N64 / TILE, M64 / TILE, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const bool col_zero_sr =
        mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_COL_ZERO_SR");
    const bool col_full_sr =
        mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_COL_DATA_SR");
    const bool col_direct_zero_sr =
        mxfp4_env_flag("FP4_CCE_V4_MXFP4_G_COL_ZERO_SR_DIRECT");
    TORCH_CHECK(
        !(col_zero_sr && col_full_sr),
        "MXFP4 full and zero-only column stochastic rounding are mutually exclusive");
    TORCH_CHECK(
        !col_direct_zero_sr || col_zero_sr,
        "direct MXFP4 zero rescue requires zero-only column SR");
    const int col_requant_rows = static_cast<int>(mxfp4_env_u64(
        "FP4_CCE_V4_MXFP4_G_COL_REQUANT_THREADS", 128));
    TORCH_CHECK(
        col_requant_rows == 32 || col_requant_rows == 64 ||
            col_requant_rows == 128 || col_requant_rows == 256,
        "FP4_CCE_V4_MXFP4_G_COL_REQUANT_THREADS must be 32, 64, 128, or 256");
    const uint64_t rng_seed =
        mxfp4_env_u64("FP4_CCE_V4_MXFP4_G_RNG_SEED", 0);
    const uint64_t rng_subsequence_base =
        mxfp4_env_u64("FP4_CCE_V4_MXFP4_G_RNG_SUBSEQUENCE_BASE", 0) +
        mxfp4_env_u64("RANK", 0) * MXFP4_SR_RANK_STRIDE;
    auto rng_state = (col_zero_sr || col_full_sr)
        ? make_mxfp4_advancing_rng_state(
            row_fp4, rng_seed, rng_subsequence_base, stream)
        : torch::Tensor();
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
#define LAUNCH_MXFP4_COL_REQUANT_NARROW(                                      \
    ROW_TILES, ROWS_PER_OUTPUT, COL_ZERO, COL_FULL, DIRECT_ZERO)               \
    mxfp4_col_requant_from_row_kernel<                                         \
        ROW_TILES, ROWS_PER_OUTPUT, true, COL_ZERO, COL_FULL, DIRECT_ZERO>     \
        <<<dim3(N64 / TILE, M64 / (ROW_TILES * ROWS_PER_OUTPUT)),              \
           ROW_TILES * TILE, 0, stream>>>(                                     \
        reinterpret_cast<const __nv_fp4x2_e2m1*>(row_fp4.data_ptr()),           \
        reinterpret_cast<const uint8_t*>(row_sc.data_ptr()),                    \
        row_normalization.data_ptr<float>(),                                    \
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),                 \
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),                          \
        static_cast<int>(M64), static_cast<int>(N64),                           \
        static_cast<float>(scale_floor_ratio), rng_seed,                        \
        rng_subsequence_base, rng_state_ptr)

#define DISPATCH_MXFP4_COL_REQUANT(ROW_TILES, ROWS_PER_OUTPUT)                \
    do {                                                                       \
        if (col_zero_sr) {                                                     \
            if (col_direct_zero_sr) {                                          \
                LAUNCH_MXFP4_COL_REQUANT_NARROW(                              \
                    ROW_TILES, ROWS_PER_OUTPUT, true, false, true);            \
            } else {                                                           \
                LAUNCH_MXFP4_COL_REQUANT_NARROW(                              \
                    ROW_TILES, ROWS_PER_OUTPUT, true, false, false);           \
            }                                                                  \
        } else if (col_full_sr) {                                              \
            LAUNCH_MXFP4_COL_REQUANT_NARROW(                                  \
                ROW_TILES, ROWS_PER_OUTPUT, false, true, false);               \
        } else {                                                               \
            LAUNCH_MXFP4_COL_REQUANT_NARROW(                                  \
                ROW_TILES, ROWS_PER_OUTPUT, false, false, false);              \
        }                                                                      \
    } while (false)
    if (col_requant_rows == 32) {
        DISPATCH_MXFP4_COL_REQUANT(1, 32);
    } else if (col_requant_rows == 64) {
        DISPATCH_MXFP4_COL_REQUANT(1, 64);
    } else if (col_requant_rows == 128) {
        DISPATCH_MXFP4_COL_REQUANT(1, 128);
    } else {
        DISPATCH_MXFP4_COL_REQUANT(2, 128);
    }
#undef DISPATCH_MXFP4_COL_REQUANT
#undef LAUNCH_MXFP4_COL_REQUANT_NARROW
    auto error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "mxfp4_col_requant_from_row failed: ", cudaGetErrorString(error));
    return std::make_tuple(col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_grad_quant_row_col_tiled(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(lse.is_cuda() && lse.is_contiguous() && lse.scalar_type() == torch::kFloat, "lse must be contiguous CUDA float");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64, "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool, "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(lse.numel() == M64 && targets.numel() == M64 && valid.numel() == M64,
                "lse/target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));

    dim3 grid(N64 / TILE, M64 / TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    mxfp4_softmax_quant_row_col_kernel<false, true><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        lse.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        valid.data_ptr<bool>(),
        reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        static_cast<int>(M64),
        static_cast<int>(N64),
        static_cast<int>(vocab_size),
        mxfp8_logit_temperature(),
        1.0f,
        0,
        0,
        nullptr);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_softmax_grad_quant_row_col_tiled failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_quant_row_col_staged(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    bool stage_exp,
    bool constant_scale) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64, "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool, "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int row_threads = select_staged_row_threads(std::getenv("FP4_CCE_V4_MXFP4_STAGED_ROW_THREADS"));
    if (row_threads == 1024) {
        launch_mxfp4_softmax_row_quant_staged<1024>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            stage_exp,
            constant_scale,
            false,
            stream);
    } else if (row_threads == 512) {
        launch_mxfp4_softmax_row_quant_staged<512>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            stage_exp,
            constant_scale,
            false,
            stream);
    } else {
        launch_mxfp4_softmax_row_quant_staged<ROW_THREADS>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            stage_exp,
            constant_scale,
            false,
            stream);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_softmax_row_quant failed: ", cudaGetErrorString(err));

    dim3 grid(N64 / TILE, M64 / TILE);
    if (constant_scale) {
        mxfp4_col_quant_from_probs_kernel<true><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64));
    } else {
        mxfp4_col_quant_from_probs_kernel<false><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            static_cast<int>(M64),
            static_cast<int>(N64));
    }
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_col_quant_from_probs failed: ", cudaGetErrorString(err));

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(loss.reshape({}), row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_softmax_grad_quant_row_col_staged(
    torch::Tensor logits,
    torch::Tensor targets,
    torch::Tensor valid,
    int64_t vocab_size,
    bool stage_exp) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be contiguous CUDA");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16 && logits.dim() == 2, "logits must be BF16 [M, N]");
    TORCH_CHECK(targets.is_cuda() && targets.is_contiguous() && targets.scalar_type() == torch::kInt64, "targets must be contiguous CUDA int64");
    TORCH_CHECK(valid.is_cuda() && valid.is_contiguous() && valid.scalar_type() == torch::kBool, "valid must be contiguous CUDA bool");

    const int64_t M64 = logits.size(0);
    const int64_t N64 = logits.size(1);
    TORCH_CHECK(M64 % TILE == 0 && N64 % TILE == 0, "M and N must be multiples of 128");
    TORCH_CHECK(targets.numel() == M64 && valid.numel() == M64, "target/valid length mismatch");
    TORCH_CHECK(vocab_size >= 0 && vocab_size <= N64, "invalid vocab_size");

    auto device = logits.device();
    auto grad_probs = torch::empty({M64, N64}, torch::dtype(torch::kBFloat16).device(device));
    auto row_fp4 = torch::empty({M64, N64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M64 / TILE, N64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({N64, M64 / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({N64 / TILE, M64 / TILE, 32, 16}, torch::dtype(torch::kUInt8).device(device));
    auto loss_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto count_sum = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int row_threads = select_staged_row_threads(std::getenv("FP4_CCE_V4_MXFP4_STAGED_ROW_THREADS"));
    if (row_threads == 1024) {
        launch_mxfp4_softmax_row_quant_staged<1024>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            stage_exp,
            false,
            true,
            stream);
    } else if (row_threads == 512) {
        launch_mxfp4_softmax_row_quant_staged<512>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            stage_exp,
            false,
            true,
            stream);
    } else {
        launch_mxfp4_softmax_row_quant_staged<ROW_THREADS>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
            targets.data_ptr<int64_t>(),
            valid.data_ptr<bool>(),
            reinterpret_cast<__nv_bfloat16*>(grad_probs.data_ptr()),
            reinterpret_cast<__nv_fp4x2_e2m1*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            loss_sum.data_ptr<float>(),
            count_sum.data_ptr<float>(),
            static_cast<int>(M64),
            static_cast<int>(N64),
            static_cast<int>(vocab_size),
            stage_exp,
            false,
            true,
            stream);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_softmax_grad_row_quant failed: ", cudaGetErrorString(err));

    dim3 grid(N64 / TILE, M64 / TILE);
    mxfp4_col_quant_from_probs_kernel<false><<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_probs.data_ptr()),
        reinterpret_cast<__nv_fp4x2_e2m1*>(col_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        static_cast<int>(M64),
        static_cast<int>(N64));
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_grad_col_quant_from_probs failed: ", cudaGetErrorString(err));

    auto loss = loss_sum / torch::clamp_min(count_sum, 1.0);
    return std::make_tuple(loss.reshape({}), row_fp4, row_sc, col_fp4, col_sc);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_softmax_quant_row_col", &mxfp4_softmax_quant_row_col,
          "Compute softmax(logits) and directly emit row/col MXFP4 P caches");
    m.def(
        "mxfp4_softmax_quant_row_col_with_floor_ratio",
        &mxfp4_softmax_quant_row_col_with_floor_ratio,
        "Compute row/col MXFP4 caches with guarded E8M0 floor rounding");
    m.def(
        "mxfp4_col_requant_from_row",
        &mxfp4_col_requant_from_row,
        "Transpose and requantize an MXFP4 row cache with delayed normalization");
    m.def("mxfp4_softmax_grad_quant_row_col_tiled", &mxfp4_softmax_grad_quant_row_col_tiled,
          "Compute G=(softmax(logits)-onehot) from precomputed LSE and directly emit row/col MXFP4 G caches");
    m.def("mxfp4_softmax_quant_row_col_staged", &mxfp4_softmax_quant_row_col_staged,
          "Compute softmax/loss plus row MXFP4, then col MXFP4 from staged BF16 probabilities");
    m.def("mxfp4_softmax_grad_quant_row_col_staged", &mxfp4_softmax_grad_quant_row_col_staged,
          "Compute softmax/loss and directly emit row/col MXFP4 G=(P-onehot) caches");
}
