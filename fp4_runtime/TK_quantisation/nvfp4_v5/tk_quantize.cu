/*
 * TK v5 — Hybrid dispatch layer + TMA scale output experiment
 *
 * Three-tier strategy for tk_quantize_for_gemm:
 *   1. grid ≤ max_concurrent → v3 fused single-pass (data stays in SMEM)
 *   2. grid > max_concurrent AND grid < persistent_threshold → v2 two-pass
 *   3. grid ≥ persistent_threshold → v5 persistent kernel (TMA scales)
 *
 * v5 change: persistent kernel uses TMA bulk stores for scales instead of
 * byte-level scattered GMEM writes.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <array>
#include <cctype>
#include <cmath>
#include <deque>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#define TK_STANDALONE
#include "core.cuh"
#include "quantize_transpose_tuned.cuh"
#define TK_QUANTIZE_TRANSPOSE_KERNEL_ONLY
#include "../nvfp4_v3/quantize_transpose.cuh"
#undef TK_QUANTIZE_TRANSPOSE_KERNEL_ONLY
#include "fused_amax_quantize.cuh"
#include "fused_group_amax_quantize.cuh"
#include "persistent_quantize.cuh"
#include "persistent_quantize_col_only.cuh"
#include "persistent_group_quantize.cuh"
#include "amax_pipelined.cuh"
#include "fused_norm_quantize.cuh"
#include "persistent_norm_quantize.cuh"
#include "persistent_gated_group_rmsnorm_quantize.cuh"
#include "localcta_h_tile_quantize.cuh"
#include "h_tile_backward.cuh"
#include "fused_silu_quantize.cuh"
#include "persistent_silu_quantize.cuh"
#include "fused_silu_deriv_quantize.cuh"
#include "persistent_silu_deriv_quantize.cuh"
#include "persistent_silu_deriv_quantize_slim.cuh"
#include "persistent_silu_deriv_gload_quantize.cuh"
#include "persistent_silu_deriv_single_pass_quantize.cuh"
#include "fused_silu_deriv_quantize_v6.cuh"
#include "persistent_sqrelu_deriv_quantize.cuh"
#include "silu_deriv_interleaved_bf16.cuh"
#include "silu_split_bf16.cuh"
#include "group_quantize_transpose.cuh"
#include "group_quantize_transpose_dim_1.cuh"
#include "../../ThunderKittens/kernels/gemm/common/c1_rms_reduce.cuh"

namespace grp_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_kernel;
namespace grp_dim1_kernel = transformer_engine::dispatch::nvfp4::group_quantize_transpose_dim1_kernel;

using namespace transformer_engine::dispatch::nvfp4::quantize_transpose_tuned_kernel;
using namespace transformer_engine;

static constexpr int kMaxPersistentLaunchDevices = 32;
static constexpr float kDefaultNvfp4GlobalScaleTarget = 448.0f;
// Above 512, a max-valued block's E4M3 multiplier rounds to zero.
static constexpr float kMaxNvfp4GlobalScaleTarget = 512.0f;

static inline void v5_check_cuda(cudaError_t status, const char* operation) {
    TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}

namespace {

struct PersistentLaunchChain {
    std::mutex mutex;
    std::deque<cudaEvent_t> pending_handoffs;
    std::vector<cudaEvent_t> available_handoffs;
    bool has_last_stream = false;
    cudaStream_t last_stream = nullptr;
};

std::array<PersistentLaunchChain, kMaxPersistentLaunchDevices> persistent_launch_chains;

thread_local int v5_quant_sequence_depth = 0;
thread_local int v5_quant_sequence_device = -1;
thread_local cudaStream_t v5_quant_sequence_stream = nullptr;

class V5QuantSequenceGuard {
public:
    explicit V5QuantSequenceGuard(cudaStream_t stream) {
        int device = -1;
        auto err = cudaGetDevice(&device);
        TORCH_CHECK(err == cudaSuccess,
                    "v5 quant sequence device query failed: ", cudaGetErrorString(err));
        TORCH_CHECK(device >= 0 && device < kMaxPersistentLaunchDevices,
                    "v5 quant sequence device is out of range: ", device);

        if (v5_quant_sequence_depth != 0) {
            TORCH_CHECK(device == v5_quant_sequence_device &&
                        stream == v5_quant_sequence_stream,
                        "nested v5 quant sequence changed device or stream");
            ++v5_quant_sequence_depth;
            return;
        }

        auto& chain = persistent_launch_chains[device];
        lock_ = std::unique_lock<std::mutex>(chain.mutex);
        if (!chain.has_last_stream || chain.last_stream != stream) {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            err = cudaStreamIsCapturing(stream, &capture_status);
            TORCH_CHECK(err == cudaSuccess,
                        "v5 quant sequence capture query failed: ", cudaGetErrorString(err));
            if (capture_status != cudaStreamCaptureStatusNone) {
                lock_.unlock();
                v5_quant_sequence_depth = 1;
                v5_quant_sequence_device = device;
                v5_quant_sequence_stream = stream;
                return;
            }
            if (chain.has_last_stream && chain.last_stream != stream) {
                while (!chain.pending_handoffs.empty()) {
                    cudaEvent_t candidate = chain.pending_handoffs.front();
                    err = cudaEventQuery(candidate);
                    if (err == cudaErrorNotReady) {
                        break;
                    }
                    TORCH_CHECK(err == cudaSuccess,
                                "v5 quant handoff query failed: ", cudaGetErrorString(err));
                    chain.pending_handoffs.pop_front();
                    chain.available_handoffs.push_back(candidate);
                }

                cudaEvent_t handoff = nullptr;
                if (chain.available_handoffs.empty()) {
                    err = cudaEventCreateWithFlags(&handoff, cudaEventDisableTiming);
                    TORCH_CHECK(err == cudaSuccess,
                                "v5 quant handoff creation failed: ", cudaGetErrorString(err));
                } else {
                    handoff = chain.available_handoffs.back();
                    chain.available_handoffs.pop_back();
                }

                err = cudaEventRecord(handoff, chain.last_stream);
                if (err != cudaSuccess) {
                    chain.available_handoffs.push_back(handoff);
                }
                TORCH_CHECK(err == cudaSuccess,
                            "v5 quant handoff record failed: ", cudaGetErrorString(err));
                chain.pending_handoffs.push_back(handoff);
                err = cudaStreamWaitEvent(stream, handoff, 0);
                TORCH_CHECK(err == cudaSuccess,
                            "v5 quant handoff wait failed: ", cudaGetErrorString(err));
            }
            chain.has_last_stream = true;
            chain.last_stream = stream;
        }

        v5_quant_sequence_depth = 1;
        v5_quant_sequence_device = device;
        v5_quant_sequence_stream = stream;
    }

    V5QuantSequenceGuard(const V5QuantSequenceGuard&) = delete;
    V5QuantSequenceGuard& operator=(const V5QuantSequenceGuard&) = delete;

    ~V5QuantSequenceGuard() {
        TORCH_INTERNAL_ASSERT(v5_quant_sequence_depth > 0);
        --v5_quant_sequence_depth;
        if (v5_quant_sequence_depth == 0) {
            v5_quant_sequence_device = -1;
            v5_quant_sequence_stream = nullptr;
        }
    }

private:
    std::unique_lock<std::mutex> lock_;
};

static inline void v5_require_tensor_device(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const char* name
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
}

static inline void v5_require_tensor_vector_device(
    const std::vector<torch::Tensor>& tensors,
    const torch::Device& device,
    const char* name
) {
    for (size_t index = 0; index < tensors.size(); ++index) {
        TORCH_CHECK(tensors[index].is_cuda(), name, "[", index, "] must be a CUDA tensor");
        TORCH_CHECK(tensors[index].device() == device,
                    name, "[", index, "] must be on ", device);
    }
}

template <typename DirectLaunch, typename CooperativeLaunch>
cudaError_t launch_barrier_grid_stream_safe(
    cudaStream_t stream,
    DirectLaunch direct_launch,
    CooperativeLaunch cooperative_launch
) {
    (void)stream;
    const char* mode = std::getenv("USE_TK_QUANT_BARRIER_LAUNCH");
    if (mode != nullptr && mode[0] != '\0') {
        if (std::strcmp(mode, "direct") == 0) {
            return direct_launch();
        }
        if (std::strcmp(mode, "cooperative") == 0) {
            return cooperative_launch();
        }
        TORCH_CHECK(false,
                    "USE_TK_QUANT_BARRIER_LAUNCH must be direct or cooperative, got ",
                    mode);
    }

    // Cooperative kernels cannot make progress while a persistent NCCL kernel
    // owns GPU resources. Distributed launchers reserve SM capacity explicitly,
    // so use an undersubscribed direct grid in that case. Standalone execution
    // keeps the cooperative launch, which guarantees whole-grid residency.
    const char* reserved_sms = std::getenv("USE_TK_QUANT_RESERVED_SMS");
    if (reserved_sms != nullptr && reserved_sms[0] != '\0' &&
        std::strtol(reserved_sms, nullptr, 10) > 0) {
        return direct_launch();
    }
    return cooperative_launch();
}

}  // namespace

template <typename Kernel, typename... Args>
static void launch_manual_barrier_kernel_capture_safe(
    const char* name,
    Kernel kernel,
    dim3 grid,
    dim3 block,
    size_t dynamic_smem,
    cudaStream_t stream,
    Args... args
) {
    const auto err = launch_barrier_grid_stream_safe(
        stream,
        [&]() {
            kernel<<<grid, block, dynamic_smem, stream>>>(args...);
            return cudaGetLastError();
        },
        [&]() {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeCooperative;
            attr.val.cooperative = 1;
            cudaLaunchConfig_t config{};
            config.gridDim = grid;
            config.blockDim = block;
            config.dynamicSmemBytes = dynamic_smem;
            config.stream = stream;
            config.attrs = &attr;
            config.numAttrs = 1;
            return cudaLaunchKernelEx(&config, kernel, args...);
        });
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ",
                cudaGetErrorString(err));
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC>
static cudaError_t launch_persistent_quantize_stream_safe(
    const CUtensorMap& tmap_in,
    const CUtensorMap& tmap_out,
    const CUtensorMap& tmap_out_t,
    const CUtensorMap& tmap_sc_row,
    const CUtensorMap& tmap_sc_col,
    nvfp4_scale_t* sc_ptr,
    int64_t M,
    int64_t K,
    int64_t scale_stride,
    tk_v5::PersistentArgs pargs,
    int num_persistent,
    int dynamic_smem,
    cudaStream_t stream
) {
    return launch_barrier_grid_stream_safe(
        stream,
        [&]() {
            tk_v5::persistent_quantize_kernel<RETURN_TRANSPOSE, ENCODE_CENTRIC>
                <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            return cudaGetLastError();
        },
        [&]() {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeCooperative;
            attr.val.cooperative = 1;
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(num_persistent);
            config.blockDim = dim3(tk_v3::V3_THREADS);
            config.dynamicSmemBytes = dynamic_smem;
            config.stream = stream;
            config.attrs = &attr;
            config.numAttrs = 1;
            return cudaLaunchKernelEx(
                &config,
                tk_v5::persistent_quantize_kernel<RETURN_TRANSPOSE, ENCODE_CENTRIC>,
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc_ptr, M, K, scale_stride, pargs);
        });
}

template <bool DIM1, bool DATA_STOCHASTIC_ROUNDING = false,
          bool COL_DATA_STOCHASTIC_ROUNDING = DATA_STOCHASTIC_ROUNDING>
static cudaError_t launch_persistent_group_quantize_stream_safe(
    const CUtensorMap& tmap_in,
    const CUtensorMap& tmap_out,
    const CUtensorMap& tmap_out_t,
    nvfp4_scale_t* scales_ptr,
    int64_t rows,
    int64_t cols,
    int64_t scale_stride,
    int64_t scale_stride_t,
    tk_v5::PersistentGroupArgs pargs,
    int num_persistent,
    int dynamic_smem,
    cudaStream_t stream
) {
    return launch_barrier_grid_stream_safe(
        stream,
        [&]() {
            if constexpr (DIM1) {
                tk_v5::persistent_group_quantize_kernel_dim1<
                    true, DATA_STOCHASTIC_ROUNDING, true,
                    COL_DATA_STOCHASTIC_ROUNDING>
                    <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                        tmap_in, tmap_out, tmap_out_t, scales_ptr,
                        rows, cols, scale_stride, scale_stride_t, pargs);
            } else {
                tk_v5::persistent_group_quantize_kernel_dim0<true>
                    <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                        tmap_in, tmap_out, tmap_out_t, scales_ptr,
                        rows, cols, scale_stride, scale_stride_t, pargs);
            }
            return cudaGetLastError();
        },
        [&]() {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeCooperative;
            attr.val.cooperative = 1;
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(num_persistent);
            config.blockDim = dim3(tk_v3::V3_THREADS);
            config.dynamicSmemBytes = dynamic_smem;
            config.stream = stream;
            config.attrs = &attr;
            config.numAttrs = 1;
            if constexpr (DIM1) {
                return cudaLaunchKernelEx(
                    &config,
                    tk_v5::persistent_group_quantize_kernel_dim1<
                        true, DATA_STOCHASTIC_ROUNDING, true,
                        COL_DATA_STOCHASTIC_ROUNDING>,
                    tmap_in, tmap_out, tmap_out_t, scales_ptr,
                    rows, cols, scale_stride, scale_stride_t, pargs);
            }
            return cudaLaunchKernelEx(
                &config,
                tk_v5::persistent_group_quantize_kernel_dim0<true>,
                tmap_in, tmap_out, tmap_out_t, scales_ptr,
                rows, cols, scale_stride, scale_stride_t, pargs);
        });
}

template <bool DIM1, bool DATA_STOCHASTIC_ROUNDING = false,
          bool COL_DATA_STOCHASTIC_ROUNDING = DATA_STOCHASTIC_ROUNDING>
static cudaError_t launch_persistent_group_quantize_two_pass(
    const CUtensorMap& tmap_in,
    const CUtensorMap& tmap_out,
    const CUtensorMap& tmap_out_t,
    nvfp4_scale_t* scales_ptr,
    int64_t rows,
    int64_t cols,
    int64_t scale_stride,
    int64_t scale_stride_t,
    tk_v5::PersistentGroupArgs pargs,
    int num_persistent,
    int dynamic_smem,
    cudaStream_t stream
) {
    if constexpr (DIM1) {
        tk_v5::persistent_group_quantize_kernel_dim1<
            true, false, true, false, 1>
            <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                tmap_in, tmap_out, tmap_out_t, scales_ptr,
                rows, cols, scale_stride, scale_stride_t, pargs);
    } else {
        tk_v5::persistent_group_quantize_kernel_dim0<true, true, 1>
            <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                tmap_in, tmap_out, tmap_out_t, scales_ptr,
                rows, cols, scale_stride, scale_stride_t, pargs);
    }
    auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    if constexpr (DIM1) {
        tk_v5::persistent_group_quantize_kernel_dim1<
            true, DATA_STOCHASTIC_ROUNDING, true,
            COL_DATA_STOCHASTIC_ROUNDING, 2>
            <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                tmap_in, tmap_out, tmap_out_t, scales_ptr,
                rows, cols, scale_stride, scale_stride_t, pargs);
    } else {
        tk_v5::persistent_group_quantize_kernel_dim0<true, true, 2>
            <<<num_persistent, tk_v3::V3_THREADS, dynamic_smem, stream>>>(
                tmap_in, tmap_out, tmap_out_t, scales_ptr,
                rows, cols, scale_stride, scale_stride_t, pargs);
    }
    return cudaGetLastError();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_transpose(
    torch::Tensor input, torch::Tensor amax_row, torch::Tensor amax_col, bool return_transpose,
    bool stochastic_rounding,
    bool encode_centric = true,
    bool scale_stochastic_rounding = false,
    bool row_rht = false,
    bool col_rht = false,
    bool return_row = true,
    torch::Tensor rng_state = torch::Tensor()
);

static constexpr unsigned long long kTkSrInvocationStride = 1ull << 32;
__device__ unsigned long long tk_sr_invocation_offset = 0;

__global__ void tk_prepare_advancing_rng_state_kernel(
    unsigned long long *rng_state,
    unsigned long long rng_seed,
    unsigned long long rng_subsequence,
    unsigned long long invocation_count
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long offset =
            atomicAdd(
                &tk_sr_invocation_offset,
                kTkSrInvocationStride * invocation_count);
        rng_state[0] = rng_seed;
        rng_state[1] = rng_subsequence + offset;
    }
}

__global__ void tk_advance_rng_state_kernel(unsigned long long *rng_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        rng_state[1] += kTkSrInvocationStride;
    }
}

static torch::Tensor tk_make_advancing_rng_state(
    const torch::Tensor &input,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    cudaStream_t stream,
    uint64_t invocation_count = 1
) {
    TORCH_CHECK(invocation_count > 0, "invocation_count must be positive");
    auto rng_state = torch::empty(
        {2}, torch::dtype(torch::kInt64).device(input.device()));
    tk_prepare_advancing_rng_state_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long *>(rng_state.data_ptr<int64_t>()),
        static_cast<unsigned long long>(rng_seed),
        static_cast<unsigned long long>(rng_subsequence),
        static_cast<unsigned long long>(invocation_count));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rng_state;
}

__device__ __forceinline__ uint32_t tk_splitmix32(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return static_cast<uint32_t>((x ^ (x >> 31)) >> 32);
}

__device__ __forceinline__ void tk_fwht16(float (&v)[16]) {
    #pragma unroll
    for (int step = 1; step < 16; step <<= 1) {
        #pragma unroll
        for (int base = 0; base < 16; base += 2 * step) {
            #pragma unroll
            for (int j = 0; j < step; ++j) {
                const float a = v[base + j];
                const float b = v[base + j + step];
                v[base + j] = a + b;
                v[base + j + step] = a - b;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] *= 0.25f;
    }
}

__device__ __forceinline__ void tk_block_reduce_max_pair(float &a, float &b, float *warp_smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    a = pipelined_amax::warp_reduce_max(a);
    b = pipelined_amax::warp_reduce_max(b);
    if (lane == 0) {
        warp_smem[warp_id] = a;
        warp_smem[pipelined_amax::WARPS + warp_id] = b;
    }
    __syncthreads();

    if (warp_id == 0) {
        a = (lane < pipelined_amax::WARPS) ? warp_smem[lane] : 0.0f;
        b = (lane < pipelined_amax::WARPS) ? warp_smem[pipelined_amax::WARPS + lane] : 0.0f;
        #pragma unroll
        for (int mask = pipelined_amax::WARPS / 2; mask > 0; mask >>= 1) {
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, mask));
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, mask));
        }
    }
}

__device__ __forceinline__ void tk_block_reduce_max_quad(
    float &a,
    float &b,
    float &c,
    float &d,
    float *warp_smem
) {
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    a = pipelined_amax::warp_reduce_max(a);
    b = pipelined_amax::warp_reduce_max(b);
    c = pipelined_amax::warp_reduce_max(c);
    d = pipelined_amax::warp_reduce_max(d);
    if (lane == 0) {
        warp_smem[warp_id] = a;
        warp_smem[pipelined_amax::WARPS + warp_id] = b;
        warp_smem[2 * pipelined_amax::WARPS + warp_id] = c;
        warp_smem[3 * pipelined_amax::WARPS + warp_id] = d;
    }
    __syncthreads();

    if (warp_id == 0) {
        a = (lane < pipelined_amax::WARPS) ? warp_smem[lane] : 0.0f;
        b = (lane < pipelined_amax::WARPS) ? warp_smem[pipelined_amax::WARPS + lane] : 0.0f;
        c = (lane < pipelined_amax::WARPS) ? warp_smem[2 * pipelined_amax::WARPS + lane] : 0.0f;
        d = (lane < pipelined_amax::WARPS) ? warp_smem[3 * pipelined_amax::WARPS + lane] : 0.0f;
        #pragma unroll
        for (int mask = pipelined_amax::WARPS / 2; mask > 0; mask >>= 1) {
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, mask));
            b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, mask));
            c = fmaxf(c, __shfl_xor_sync(0xffffffff, c, mask));
            d = fmaxf(d, __shfl_xor_sync(0xffffffff, d, mask));
        }
    }
}

__global__ void rht16_bf16_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ amax_out,
    float* __restrict__ sg_out,
    int64_t M,
    int64_t K,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    extern __shared__ float warp_smem[];
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_max = 0.0f;

    for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < groups;
         group += stride) {
        float vals[16];
        int64_t base = 0;
        int64_t step = 1;
        int64_t sign_coord_base = 0;
        if (across_cols) {
            const int64_t blocks_per_row = K / 16;
            const int64_t row = group / blocks_per_row;
            const int64_t col_block = group - row * blocks_per_row;
            base = row * K + col_block * 16;
            step = 1;
            sign_coord_base = col_block * 16;
        } else {
            const int64_t row_block = group / K;
            const int64_t col = group - row_block * K;
            base = row_block * 16 * K + col;
            step = K;
            sign_coord_base = row_block * 16;
        }

        if (across_cols && !with_random_sign_mask) {
            #pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float2 x = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(input + base + i));
                vals[i] = x.x;
                vals[i + 1] = x.y;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                float x = __bfloat162float(input[base + i * step]);
                if (with_random_sign_mask) {
                    const uint64_t h = rng_seed
                        ^ (rng_subsequence * 0xd1b54a32d192ed03ull)
                        ^ (static_cast<uint64_t>(sign_coord_base + i) * 0x9e3779b97f4a7c15ull);
                    x = (tk_splitmix32(h) & 1u) ? x : -x;
                }
                vals[i] = x;
            }
        }

        tk_fwht16(vals);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float x = vals[i];
            local_max = fmaxf(local_max, fabsf(x));
            if (output != nullptr) {
                output[base + i * step] = __float2bfloat16(x);
            }
        }
    }

    const float block_max = pipelined_amax::block_reduce_max(local_max, warp_smem);
    if (threadIdx.x == 0) {
        pipelined_amax::atomic_max_float(amax_out, block_max);
    }

    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int rht_amax_block_count = 0;
        const unsigned int prev = atomicAdd(&rht_amax_block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) rht_amax_block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        *sg_out = *amax_out / 2688.0f;
    }
}

__global__ void rht16_bf16_amax_with_orig_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ rht_amax_out,
    float* __restrict__ rht_sg_out,
    float* __restrict__ orig_amax_out,
    float* __restrict__ orig_sg_out,
    int64_t M,
    int64_t K,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    extern __shared__ float warp_smem[];
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_rht_max = 0.0f;
    float local_orig_max = 0.0f;

    for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < groups;
         group += stride) {
        float vals[16];
        int64_t base = 0;
        int64_t step = 1;
        int64_t sign_coord_base = 0;
        if (across_cols) {
            const int64_t blocks_per_row = K / 16;
            const int64_t row = group / blocks_per_row;
            const int64_t col_block = group - row * blocks_per_row;
            base = row * K + col_block * 16;
            step = 1;
            sign_coord_base = col_block * 16;
        } else {
            const int64_t row_block = group / K;
            const int64_t col = group - row_block * K;
            base = row_block * 16 * K + col;
            step = K;
            sign_coord_base = row_block * 16;
        }

        if (across_cols && !with_random_sign_mask) {
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float2 x = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(input + base + i));
                vals[i] = x.x;
                vals[i + 1] = x.y;
                local_orig_max = fmaxf(local_orig_max, fabsf(x.x));
                local_orig_max = fmaxf(local_orig_max, fabsf(x.y));
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float x = __bfloat162float(input[base + i * step]);
                local_orig_max = fmaxf(local_orig_max, fabsf(x));
                if (with_random_sign_mask) {
                    const uint64_t h = rng_seed
                        ^ (rng_subsequence * 0xd1b54a32d192ed03ull)
                        ^ (static_cast<uint64_t>(sign_coord_base + i) * 0x9e3779b97f4a7c15ull);
                    x = (tk_splitmix32(h) & 1u) ? x : -x;
                }
                vals[i] = x;
            }
        }

        tk_fwht16(vals);

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float x = vals[i];
            local_rht_max = fmaxf(local_rht_max, fabsf(x));
            if (output != nullptr) {
                output[base + i * step] = __float2bfloat16(x);
            }
        }
    }

    tk_block_reduce_max_pair(local_rht_max, local_orig_max, warp_smem);
    if (threadIdx.x == 0) {
        pipelined_amax::atomic_max_float(rht_amax_out, local_rht_max);
        pipelined_amax::atomic_max_float(orig_amax_out, local_orig_max);
    }

    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int rht_orig_amax_block_count = 0;
        const unsigned int prev = atomicAdd(&rht_orig_amax_block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) rht_orig_amax_block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        *rht_sg_out = *rht_amax_out / 2688.0f;
        *orig_sg_out = *orig_amax_out / 2688.0f;
    }
}

__device__ __forceinline__ float tk_sqrelu_fwd_bf16_value(float x) {
    const float r = fmaxf(x, 0.0f);
    return __bfloat162float(__float2bfloat16_rn(r * r));
}

__device__ __forceinline__ float tk_sqrelu_bwd_bf16_value(float dh, float x) {
    const float r = fmaxf(x, 0.0f);
    return __bfloat162float(__float2bfloat16_rn(2.0f * dh * r));
}

__global__ void sqrelu_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ aux,
    float* __restrict__ amax_out,
    float* __restrict__ sg_out,
    int64_t n,
    bool deriv
) {
    extern __shared__ float warp_smem[];
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_max = 0.0f;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        const float x = __bfloat162float(input[idx]);
        const float y = deriv
            ? tk_sqrelu_bwd_bf16_value(x, __bfloat162float(aux[idx]))
            : tk_sqrelu_fwd_bf16_value(x);
        local_max = fmaxf(local_max, fabsf(y));
    }

    const float block_max = pipelined_amax::block_reduce_max(local_max, warp_smem);
    if (threadIdx.x == 0) {
        pipelined_amax::atomic_max_float(amax_out, block_max);
    }

    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int sqrelu_amax_block_count = 0;
        const unsigned int prev = atomicAdd(&sqrelu_amax_block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) sqrelu_amax_block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        *sg_out = *amax_out / 2688.0f;
    }
}

__global__ void sqrelu_rht16_amax_with_orig_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ aux,
    float* __restrict__ rht_amax_out,
    float* __restrict__ rht_sg_out,
    float* __restrict__ orig_amax_out,
    float* __restrict__ orig_sg_out,
    int64_t M,
    int64_t K,
    bool across_cols,
    bool deriv
) {
    extern __shared__ float warp_smem[];
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_rht_max = 0.0f;
    float local_orig_max = 0.0f;

    for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < groups;
         group += stride) {
        float vals[16];
        int64_t base = 0;
        int64_t step = 1;
        if (across_cols) {
            const int64_t blocks_per_row = K / 16;
            const int64_t row = group / blocks_per_row;
            const int64_t col_block = group - row * blocks_per_row;
            base = row * K + col_block * 16;
            step = 1;
        } else {
            const int64_t row_block = group / K;
            const int64_t col = group - row_block * K;
            base = row_block * 16 * K + col;
            step = K;
        }

        if (across_cols && !deriv) {
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float2 x = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(input + base + i));
                vals[i] = tk_sqrelu_fwd_bf16_value(x.x);
                vals[i + 1] = tk_sqrelu_fwd_bf16_value(x.y);
                local_orig_max = fmaxf(local_orig_max, fabsf(vals[i]));
                local_orig_max = fmaxf(local_orig_max, fabsf(vals[i + 1]));
            }
        } else if (across_cols && deriv) {
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const float2 dh = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(input + base + i));
                const float2 x = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(aux + base + i));
                vals[i] = tk_sqrelu_bwd_bf16_value(dh.x, x.x);
                vals[i + 1] = tk_sqrelu_bwd_bf16_value(dh.y, x.y);
                local_orig_max = fmaxf(local_orig_max, fabsf(vals[i]));
                local_orig_max = fmaxf(local_orig_max, fabsf(vals[i + 1]));
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int64_t idx = base + i * step;
                const float x = __bfloat162float(input[idx]);
                vals[i] = deriv
                    ? tk_sqrelu_bwd_bf16_value(x, __bfloat162float(aux[idx]))
                    : tk_sqrelu_fwd_bf16_value(x);
                local_orig_max = fmaxf(local_orig_max, fabsf(vals[i]));
            }
        }

        tk_fwht16(vals);

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            local_rht_max = fmaxf(local_rht_max, fabsf(vals[i]));
        }
    }

    tk_block_reduce_max_pair(local_rht_max, local_orig_max, warp_smem);
    if (threadIdx.x == 0) {
        pipelined_amax::atomic_max_float(rht_amax_out, local_rht_max);
        pipelined_amax::atomic_max_float(orig_amax_out, local_orig_max);
    }

    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int sqrelu_rht_orig_amax_block_count = 0;
        const unsigned int prev = atomicAdd(&sqrelu_rht_orig_amax_block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) sqrelu_rht_orig_amax_block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        *rht_sg_out = *rht_amax_out / 2688.0f;
        *orig_sg_out = *orig_amax_out / 2688.0f;
    }
}

__global__ void rmsnorm_rht16_amax_with_orig_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ inv_rms,
    float* __restrict__ rht_amax_out,
    float* __restrict__ rht_sg_out,
    float* __restrict__ orig_amax_out,
    float* __restrict__ orig_sg_out,
    int64_t M,
    int64_t K,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    extern __shared__ float warp_smem[];
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float local_rht_max = 0.0f;
    float local_orig_max = 0.0f;

    for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < groups;
         group += stride) {
        float vals[16];
        int64_t row_base = 0;
        int64_t col_base = 0;
        int64_t sign_coord_base = 0;
        if (across_cols) {
            const int64_t blocks_per_row = K / 16;
            const int64_t row = group / blocks_per_row;
            const int64_t col_block = group - row * blocks_per_row;
            row_base = row;
            col_base = col_block * 16;
            sign_coord_base = col_base;
        } else {
            const int64_t row_block = group / K;
            const int64_t col = group - row_block * K;
            row_base = row_block * 16;
            col_base = col;
            sign_coord_base = row_base;
        }

        if (across_cols && !with_random_sign_mask) {
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const int64_t idx = row_base * K + col_base + i;
                const float2 x = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(input + idx));
                const float2 g = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(gamma + col_base + i));
                const float inv = inv_rms[row_base];
                const float2 raw = {x.x * inv * g.x, x.y * inv * g.y};
                const float2 norm = __bfloat1622float2(__float22bfloat162_rn(raw));
                vals[i] = norm.x;
                vals[i + 1] = norm.y;
                local_orig_max = fmaxf(local_orig_max, fabsf(norm.x));
                local_orig_max = fmaxf(local_orig_max, fabsf(norm.y));
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int64_t row = across_cols ? row_base : row_base + i;
                const int64_t col = across_cols ? col_base + i : col_base;
                const int64_t idx = row * K + col;
                const float x = __bfloat162float(input[idx]);
                const float g = __bfloat162float(gamma[col]);
                float norm = __bfloat162float(__float2bfloat16_rn(x * inv_rms[row] * g));
                local_orig_max = fmaxf(local_orig_max, fabsf(norm));
                if (with_random_sign_mask) {
                    const uint64_t h = rng_seed
                        ^ (rng_subsequence * 0xd1b54a32d192ed03ull)
                        ^ (static_cast<uint64_t>(sign_coord_base + i) * 0x9e3779b97f4a7c15ull);
                    norm = (tk_splitmix32(h) & 1u) ? norm : -norm;
                }
                vals[i] = norm;
            }
        }

        tk_fwht16(vals);

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            local_rht_max = fmaxf(local_rht_max, fabsf(vals[i]));
        }
    }

    tk_block_reduce_max_pair(local_rht_max, local_orig_max, warp_smem);
    if (threadIdx.x == 0) {
        pipelined_amax::atomic_max_float(rht_amax_out, local_rht_max);
        pipelined_amax::atomic_max_float(orig_amax_out, local_orig_max);
    }

    __threadfence();
    __shared__ bool is_last;
    if (threadIdx.x == 0) {
        static __device__ unsigned int rmsnorm_rht_orig_amax_block_count = 0;
        const unsigned int prev = atomicAdd(&rmsnorm_rht_orig_amax_block_count, 1);
        is_last = (prev == gridDim.x - 1);
        if (is_last) rmsnorm_rht_orig_amax_block_count = 0;
    }
    __syncthreads();
    if (is_last && threadIdx.x == 0) {
        *rht_sg_out = *rht_amax_out / 2688.0f;
        *orig_sg_out = *orig_amax_out / 2688.0f;
    }
}

template <bool COMPUTE_INV_RMS>
__global__ void rmsnorm_bf16_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma,
    float* __restrict__ inv_rms,
    float* __restrict__ amax_out,
    float epsilon,
    int64_t M,
    int64_t K
) {
    constexpr int kBlockSize = 256;
    __shared__ float warp_values[kBlockSize / 32];
    __shared__ float row_inv_rms;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
        const int64_t row_offset = row * K;
        if constexpr (COMPUTE_INV_RMS) {
            float sum_sq = 0.0f;
            for (int64_t col = 2 * threadIdx.x; col < K;
                 col += 2 * kBlockSize) {
                const float2 x = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(
                        input + row_offset + col));
                sum_sq = fmaf(x.x, x.x, sum_sq);
                sum_sq = fmaf(x.y, x.y, sum_sq);
            }
            for (int mask = 16; mask > 0; mask >>= 1) {
                sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
            }
            if (lane == 0) warp_values[warp_id] = sum_sq;
            __syncthreads();
            if (warp_id == 0) {
                sum_sq = lane < kBlockSize / 32 ? warp_values[lane] : 0.0f;
                for (int mask = (kBlockSize / 32) / 2; mask > 0; mask >>= 1) {
                    sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
                }
                if (lane == 0) {
                    row_inv_rms = rsqrtf(sum_sq / static_cast<float>(K) + epsilon);
                    inv_rms[row] = row_inv_rms;
                }
            }
            __syncthreads();
        } else {
            if (threadIdx.x == 0) row_inv_rms = inv_rms[row];
            __syncthreads();
        }

        float local_max = 0.0f;
        for (int64_t col = 2 * threadIdx.x; col < K;
             col += 2 * kBlockSize) {
            const float2 x = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(
                    input + row_offset + col));
            const float2 g = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(gamma + col));
            const float2 raw = {
                x.x * row_inv_rms * g.x,
                x.y * row_inv_rms * g.y,
            };
            const float2 norm = __bfloat1622float2(__float22bfloat162_rn(raw));
            local_max = fmaxf(local_max, fabsf(norm.x));
            local_max = fmaxf(local_max, fabsf(norm.y));
        }

        for (int mask = 16; mask > 0; mask >>= 1) {
            local_max = fmaxf(
                local_max,
                __shfl_xor_sync(0xffffffff, local_max, mask));
        }
        if (lane == 0) warp_values[warp_id] = local_max;
        __syncthreads();
        if (warp_id == 0) {
            local_max = lane < kBlockSize / 32 ? warp_values[lane] : 0.0f;
            for (int mask = (kBlockSize / 32) / 2; mask > 0; mask >>= 1) {
                local_max = fmaxf(
                    local_max,
                    __shfl_xor_sync(0xffffffff, local_max, mask));
            }
            if (lane == 0) {
                pipelined_amax::atomic_max_float(amax_out, local_max);
            }
        }
        __syncthreads();
    }
}

__global__ void dual_bf16_row_rht_orig_amax_kernel(
    const __nv_bfloat16* __restrict__ input1,
    const __nv_bfloat16* __restrict__ input2,
    float* __restrict__ amaxes,
    int64_t M,
    int64_t K
) {
    extern __shared__ float warp_smem[];
    const int64_t groups = M * (K / 16);
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    float row_max1 = 0.0f;
    float col_max1 = 0.0f;
    float row_max2 = 0.0f;
    float col_max2 = 0.0f;

    for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         group < groups;
         group += stride) {
        const int64_t row = group / (K / 16);
        const int64_t col_block = group - row * (K / 16);
        const int64_t base = row * K + col_block * 16;
        float vals1[16];
        float vals2[16];

#pragma unroll
        for (int i = 0; i < 16; i += 2) {
            const int64_t idx = base + i;
            const float2 x1 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(input1 + idx));
            const float2 x2 = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(input2 + idx));
            vals1[i] = x1.x;
            vals1[i + 1] = x1.y;
            vals2[i] = x2.x;
            vals2[i + 1] = x2.y;
            col_max1 = fmaxf(col_max1, fabsf(x1.x));
            col_max1 = fmaxf(col_max1, fabsf(x1.y));
            col_max2 = fmaxf(col_max2, fabsf(x2.x));
            col_max2 = fmaxf(col_max2, fabsf(x2.y));
        }

        tk_fwht16(vals1);
        tk_fwht16(vals2);

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            row_max1 = fmaxf(row_max1, fabsf(vals1[i]));
            row_max2 = fmaxf(row_max2, fabsf(vals2[i]));
        }
    }

    tk_block_reduce_max_quad(row_max1, col_max1, row_max2, col_max2, warp_smem);

    if (threadIdx.x == 0) {
        if (row_max1 > 0.0f) transformer_engine::atomicMaxFloat(amaxes + 0, row_max1);
        if (col_max1 > 0.0f) transformer_engine::atomicMaxFloat(amaxes + 1, col_max1);
        if (row_max2 > 0.0f) transformer_engine::atomicMaxFloat(amaxes + 2, row_max2);
        if (col_max2 > 0.0f) transformer_engine::atomicMaxFloat(amaxes + 3, col_max2);
    }
}

// ─────────────────────── TMA tensor map creation ────────────────
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

// ═══════════════════════════════════════════════════════════════════
// Vectorized silu(h1)*h3 kernel for strided h13 layout
// Matches TE's fused_silu_mul_strided_amax_kernel: bf162 loads,
// grid-striding loop, capped grid size.
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void silu_strided_kernel(
    const __nv_bfloat16* __restrict__ h13,  // (M, 2H) contiguous
    __nv_bfloat16* __restrict__ out,        // (M, H) contiguous output
    int64_t M, int64_t H) {

    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    // Each thread processes 2 bf16 per iteration via bf162
    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;

        __nv_bfloat162 h1_val = *reinterpret_cast<const __nv_bfloat162*>(h1_ptr);
        __nv_bfloat162 h3_val = *reinterpret_cast<const __nv_bfloat162*>(h3_ptr);

        float2 h1_f = __bfloat1622float2(h1_val);
        h1_f.x = silu_f(h1_f.x);
        h1_f.y = silu_f(h1_f.y);

        float2 h3_f = __bfloat1622float2(h3_val);

        float2 r_f;
        r_f.x = h1_f.x * h3_f.x;
        r_f.y = h1_f.y * h3_f.y;

        *reinterpret_cast<__nv_bfloat162*>(out + elem) = __float22bfloat162_rn(r_f);
    }

    // Handle odd remainder
    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float h1_f = silu_f(__bfloat162float(h13[row * row_stride + col]));
        float h3_f = __bfloat162float(h13[row * row_stride + H + col]);
        out[i] = __float2bfloat16(h1_f * h3_f);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Vectorized silu-deriv dual multiply kernel for backward pass
// Computes: out1 = dh * h3 * silu'(h1)   (gradient w.r.t. h1)
//           out2 = dh * silu(h1)          (gradient w.r.t. h3)
// silu'(x) = sigmoid(x) * (1 + x - silu(x))
// ═══════════════════════════════════════════════════════════════════
__global__ void silu_deriv_dual_strided_kernel(
    const __nv_bfloat16* __restrict__ dh,    // (M, H) contiguous
    const __nv_bfloat16* __restrict__ h13,   // (M, 2H) contiguous
    __nv_bfloat16* __restrict__ out1,        // (M, H) dh * h3 * silu'(h1)
    __nv_bfloat16* __restrict__ out2,        // (M, H) dh * silu(h1)
    int64_t M, int64_t H) {

    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        __nv_bfloat162 dh_val = *reinterpret_cast<const __nv_bfloat162*>(dh + elem);
        float2 dh_f = __bfloat1622float2(dh_val);

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        float2 h1_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h1_ptr));
        float2 h3_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h3_ptr));

        float sigx = 1.0f / (1.0f + expf(-h1_f.x));
        float sigy = 1.0f / (1.0f + expf(-h1_f.y));
        float silux = h1_f.x * sigx;
        float siluy = h1_f.y * sigy;
        float silupx = sigx * (1.0f + h1_f.x - silux);
        float silupy = sigy * (1.0f + h1_f.y - siluy);

        float2 c_f = { dh_f.x * h3_f.x * silupx, dh_f.y * h3_f.y * silupy };
        float2 e_f = { dh_f.x * silux, dh_f.y * siluy };

        *reinterpret_cast<__nv_bfloat162*>(out1 + elem) = __float22bfloat162_rn(c_f);
        *reinterpret_cast<__nv_bfloat162*>(out2 + elem) = __float22bfloat162_rn(e_f);
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float vd = __bfloat162float(dh[i]);
        float v1 = __bfloat162float(h13[row * row_stride + col]);
        float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        float sig = 1.0f / (1.0f + expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);
        out1[i] = __float2bfloat16(vd * v3 * silup_v1);
        out2[i] = __float2bfloat16(vd * silu_v1);
    }
}

// Amax-only variant: same math as silu_deriv_dual_strided_kernel but
// only computes atomicMax of abs values → no bf16 output writes.
// Used as Phase 1 of the slim 2-kernel path.
__global__ void silu_deriv_amax_only_kernel(
    const __nv_bfloat16* __restrict__ dh,    // (M, H)
    const __nv_bfloat16* __restrict__ h13,   // (M, 2H)
    float* __restrict__ global_amax1,        // atomic max of |dh1|
    float* __restrict__ global_amax2,        // atomic max of |dh3|
    int64_t M, int64_t H) {

    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    float local_max1 = 0.0f, local_max2 = 0.0f;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        __nv_bfloat162 dh_val = *reinterpret_cast<const __nv_bfloat162*>(dh + elem);
        float2 dh_f = __bfloat1622float2(dh_val);

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        float2 h1_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h1_ptr));
        float2 h3_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h3_ptr));

        float sigx = 1.0f / (1.0f + expf(-h1_f.x));
        float sigy = 1.0f / (1.0f + expf(-h1_f.y));
        float silux = h1_f.x * sigx;
        float siluy = h1_f.y * sigy;
        float silupx = sigx * (1.0f + h1_f.x - silux);
        float silupy = sigy * (1.0f + h1_f.y - siluy);

        float cx = fabsf(dh_f.x * h3_f.x * silupx);
        float cy = fabsf(dh_f.y * h3_f.y * silupy);
        float ex = fabsf(dh_f.x * silux);
        float ey = fabsf(dh_f.y * siluy);

        local_max1 = fmaxf(local_max1, fmaxf(cx, cy));
        local_max2 = fmaxf(local_max2, fmaxf(ex, ey));
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float vd = __bfloat162float(dh[i]);
        float v1 = __bfloat162float(h13[row * row_stride + col]);
        float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        float sig = 1.0f / (1.0f + expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);
        local_max1 = fmaxf(local_max1, fabsf(vd * v3 * silup_v1));
        local_max2 = fmaxf(local_max2, fabsf(vd * silu_v1));
    }

    // Warp-level reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, offset));
        local_max2 = fmaxf(local_max2, __shfl_xor_sync(0xffffffff, local_max2, offset));
    }

    // One thread per warp does atomicMax
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(global_amax1),
                  __float_as_uint(local_max1));
        atomicMax(reinterpret_cast<unsigned int*>(global_amax2),
                  __float_as_uint(local_max2));
    }
}

// Fused silu_deriv + dual amax: writes bf16 outputs AND computes dual amaxes
// in a single GMEM pass. Eliminates the need for phase1 amax scan in quantize.
__global__ void silu_deriv_dual_amax_kernel(
    const __nv_bfloat16* __restrict__ dh,    // (M, H)
    const __nv_bfloat16* __restrict__ h13,   // (M, 2H)
    __nv_bfloat16* __restrict__ out1,        // (M, H) dh * h3 * silu'(h1)
    __nv_bfloat16* __restrict__ out2,        // (M, H) dh * silu(h1)
    float* __restrict__ global_amax1,        // atomic max of |out1|
    float* __restrict__ global_amax2,        // atomic max of |out2|
    int64_t M, int64_t H) {

    const int64_t total = M * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t row_stride = 2 * H;

    float local_max1 = 0.0f, local_max2 = 0.0f;

    for (int64_t i = idx; i < total / 2; i += stride) {
        int64_t elem = i * 2;
        int64_t row = elem / H;
        int64_t col = elem % H;

        __nv_bfloat162 dh_val = *reinterpret_cast<const __nv_bfloat162*>(dh + elem);
        float2 dh_f = __bfloat1622float2(dh_val);

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        float2 h1_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h1_ptr));
        float2 h3_f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h3_ptr));

        float sigx = 1.0f / (1.0f + expf(-h1_f.x));
        float sigy = 1.0f / (1.0f + expf(-h1_f.y));
        float silux = h1_f.x * sigx;
        float siluy = h1_f.y * sigy;
        float silupx = sigx * (1.0f + h1_f.x - silux);
        float silupy = sigy * (1.0f + h1_f.y - siluy);

        float2 c_f = { dh_f.x * h3_f.x * silupx, dh_f.y * h3_f.y * silupy };
        float2 e_f = { dh_f.x * silux, dh_f.y * siluy };

        *reinterpret_cast<__nv_bfloat162*>(out1 + elem) = __float22bfloat162_rn(c_f);
        *reinterpret_cast<__nv_bfloat162*>(out2 + elem) = __float22bfloat162_rn(e_f);

        local_max1 = fmaxf(local_max1, fmaxf(fabsf(c_f.x), fabsf(c_f.y)));
        local_max2 = fmaxf(local_max2, fmaxf(fabsf(e_f.x), fabsf(e_f.y)));
    }

    if (total % 2 != 0 && idx == 0) {
        int64_t i = total - 1;
        int64_t row = i / H, col = i % H;
        float vd = __bfloat162float(dh[i]);
        float v1 = __bfloat162float(h13[row * row_stride + col]);
        float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        float sig = 1.0f / (1.0f + expf(-v1));
        float silu_v1 = v1 * sig;
        float silup_v1 = sig * (1.0f + v1 - silu_v1);
        float c = vd * v3 * silup_v1;
        float e = vd * silu_v1;
        out1[i] = __float2bfloat16(c);
        out2[i] = __float2bfloat16(e);
        local_max1 = fmaxf(local_max1, fabsf(c));
        local_max2 = fmaxf(local_max2, fabsf(e));
    }

    // Warp-level reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, offset));
        local_max2 = fmaxf(local_max2, __shfl_xor_sync(0xffffffff, local_max2, offset));
    }

    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(global_amax1),
                  __float_as_uint(local_max1));
        atomicMax(reinterpret_cast<unsigned int*>(global_amax2),
                  __float_as_uint(local_max2));
    }
}

// ═══════════════════════════════════════════════════════════════════
// v3 fused launch helper
// ═══════════════════════════════════════════════════════════════════
template <bool RT, bool ENCODE_CENTRIC = true>
static void launch_v3(
    const CUtensorMap &tmap_in, const CUtensorMap &tmap_out, const CUtensorMap &tmap_out_t,
    nvfp4_scale_t *sc_ptr, nvfp4_scale_t *sc_t_ptr,
    float *global_amax, float *sg_out,
    unsigned int *done_counter, unsigned int *ready_flag,
    int64_t M, int64_t K, int64_t scale_stride, int64_t scale_stride_t,
    cudaStream_t stream
) {
    using namespace tk_v3;
    const int blocks_Y = (M + V3Config::CHUNK_DIM_Y - 1) / V3Config::CHUNK_DIM_Y;
    const int blocks_X = (K + V3Config::CHUNK_DIM_X - 1) / V3Config::CHUNK_DIM_X;
    const int total_blocks = blocks_X * blocks_Y;
    const dim3 grid(blocks_X, blocks_Y);
    const int dshmem = v3_shmem_size<RT>();

    auto kernel = fused_amax_quantize_kernel<RT, ENCODE_CENTRIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, V3_THREADS, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        sc_ptr, sc_t_ptr, global_amax, sg_out,
        done_counter, ready_flag,
        M, K, scale_stride, scale_stride_t, total_blocks);
}

// ═══════════════════════════════════════════════════════════════════
// v2 launch helper (pipelined amax + separate quantize)
// ═══════════════════════════════════════════════════════════════════
template <bool SR, bool FM, bool RT, bool ENCODE_CENTRIC = true, bool SCALE_SR = false,
          bool ROW_RHT = false, bool COL_RHT = false, bool RETURN_ROW = true,
          bool APPLY_SQRELU = false, bool APPLY_SQRELU_DERIV = false>
static void launch_v2_kernel(
    const CUtensorMap &tmap_in, const CUtensorMap &tmap_out, const CUtensorMap &tmap_out_t,
    nvfp4_scale_t *sc_ptr, nvfp4_scale_t *sc_t_ptr,
    const float *amax_row, const float *amax_col,
    int64_t M, int64_t K, int64_t scale_stride, int64_t scale_stride_t,
    cudaStream_t stream,
    const IType *sqrelu_aux_ptr = nullptr,
    const size_t *rng_state_ptr = nullptr
) {
    const int blocks_Y = (M + TunableConfig::CHUNK_DIM_Y - 1) / TunableConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + TunableConfig::CHUNK_DIM_X - 1) / TunableConfig::CHUNK_DIM_X;
    const dim3 grid(blocks_X, blocks_Y);

    constexpr int buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
    constexpr int bsz_in  = ((BUFFS_NUM_IN * buff_elems * (int)sizeof(bf16) + 127) / 128) * 128;
    constexpr int bsz_out = RETURN_ROW ? ((BUFFS_NUM_OUT * BUFF_OUT_SIZE + 127) / 128) * 128 : 0;
    constexpr int bsz_out_t = RT ? (((BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE + 127) / 128) * 128) : 0;
    constexpr int bsz_sc  = RETURN_ROW ? ((TunableConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128 : 0;
    constexpr int bsz_sc_t = RT ? (((TunableConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128) : 0;
    constexpr int dshmem = bsz_in + bsz_out + bsz_out_t + bsz_sc + bsz_sc_t + 128;

    auto kernel =
        quantize_transpose_nvfp4_tuned_1D_kernel<SR, FM, RT, ENCODE_CENTRIC, RETURN_ROW,
                                                 ROW_RHT, COL_RHT, SCALE_SR, false,
                                                 APPLY_SQRELU, APPLY_SQRELU_DERIV>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
	    kernel<<<grid, THREADS_NUM, dshmem, stream>>>(
	        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
	        nullptr, amax_row, amax_col, M, K,
	        scale_stride, scale_stride_t, rng_state_ptr, true, nullptr, nullptr, sqrelu_aux_ptr);
}

template <bool SR, bool FM, bool RT, bool ENCODE_CENTRIC = true, bool SCALE_SR = false,
          bool ROW_RHT = false, bool COL_RHT = false, bool RETURN_ROW = true>
static void launch_v2_rmsnorm_kernel(
    const CUtensorMap &tmap_in, const CUtensorMap &tmap_out, const CUtensorMap &tmap_out_t,
    nvfp4_scale_t *sc_ptr, nvfp4_scale_t *sc_t_ptr,
    const float *amax_row, const float *amax_col,
    const float *inv_rms_ptr, const tk_v3::IType *gamma_ptr,
    int64_t M, int64_t K, int64_t scale_stride, int64_t scale_stride_t,
    cudaStream_t stream,
    const size_t *rng_state_ptr = nullptr
) {
    const int blocks_Y = (M + TunableConfig::CHUNK_DIM_Y - 1) / TunableConfig::CHUNK_DIM_Y;
    const int blocks_X = (K + TunableConfig::CHUNK_DIM_X - 1) / TunableConfig::CHUNK_DIM_X;
    const dim3 grid(blocks_X, blocks_Y);

    constexpr int buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
    constexpr int bsz_in  = ((BUFFS_NUM_IN * buff_elems * (int)sizeof(bf16) + 127) / 128) * 128;
    constexpr int bsz_out = RETURN_ROW ? ((BUFFS_NUM_OUT * BUFF_OUT_SIZE + 127) / 128) * 128 : 0;
    constexpr int bsz_out_t = RT ? (((BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE + 127) / 128) * 128) : 0;
    constexpr int bsz_sc  = RETURN_ROW ? ((TunableConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128 : 0;
    constexpr int bsz_sc_t = RT ? (((TunableConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t) + 127) / 128) * 128) : 0;
    constexpr int dshmem = bsz_in + bsz_out + bsz_out_t + bsz_sc + bsz_sc_t + 128;

    auto kernel =
        quantize_transpose_nvfp4_tuned_1D_kernel<SR, FM, RT, ENCODE_CENTRIC, RETURN_ROW,
                                                 ROW_RHT, COL_RHT, SCALE_SR, true>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    kernel<<<grid, THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr,
        nullptr, amax_row, amax_col, M, K,
        scale_stride, scale_stride_t, rng_state_ptr, true, inv_rms_ptr, gamma_ptr, nullptr);
}


// ═══════════════════════════════════════════════════════════════════
// v2 grouped launch helpers
// ═══════════════════════════════════════════════════════════════════
template <bool SR, bool RT>
static void launch_group_kernel_v2(
    const CUtensorMap &tmap_in0, const CUtensorMap &tmap_in1, const CUtensorMap &tmap_out,
    nvfp4_scale_t *sc_ptr,
    int64_t rows, int64_t cols, int64_t scale_stride,
    grp_kernel::MultiAmaxCastTransposeFusionArgs &kernel_args,
    cudaStream_t stream
) {
    const int blocks_Y = (rows + grp_kernel::CHUNK_DIM_Y - 1) / grp_kernel::CHUNK_DIM_Y;
    const int blocks_X = (cols + grp_kernel::CHUNK_DIM_X - 1) / grp_kernel::CHUNK_DIM_X;
    const dim3 grid(blocks_X, blocks_Y);

    constexpr size_t buff_elems = grp_kernel::BUFF_DIM_Y * grp_kernel::BUFF_DIM_X;
    constexpr size_t buff_elems_total = grp_kernel::BUFFS_NUM * buff_elems;
    constexpr size_t bsz_in  = ((buff_elems_total * sizeof(bf16) + 127) / 128) * 128;
    constexpr size_t bsz_out = (((buff_elems_total * 4) / 8 + 127) / 128) * 128;
    constexpr size_t bsz_out_t = RT ? bsz_out : 0;
    constexpr size_t bsz_sc_t = RT ? (((grp_kernel::CHUNK_DIM_Y * grp_kernel::CHUNK_DIM_X) / 16 * sizeof(nvfp4_scale_t) + 127) / 128) * 128 : 0;
    constexpr size_t dshmem = bsz_in + bsz_out + bsz_out_t + bsz_sc_t + 128;

    using Empty = transformer_engine::Empty;
    auto kernel = grp_kernel::group_quantize_transpose_nvfp4_kernel<false, Empty, nullptr, bf16, SR, RT>;
    // Set shmem attribute only once (first call) — skip during CUDA graph capture
    static bool s_attr_set = false;
    if (!s_attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        s_attr_set = true;
    }
    kernel<<<grid, grp_kernel::THREADS_NUM, dshmem, stream>>>(
        tmap_in0, tmap_in1, tmap_out, sc_ptr, nullptr,
        rows, cols, scale_stride, nullptr, kernel_args);
}

__global__ void compute_sg_kernel(const float* __restrict__ amaxes,
                                   float* __restrict__ sgs,
                                   int num);

static constexpr int BF16_TR_TILE = 32;

__global__ void bf16_transpose_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int M,
    int K
) {
    __shared__ __nv_bfloat16 tile[BF16_TR_TILE][BF16_TR_TILE + 1];

    int src_x = blockIdx.x * BF16_TR_TILE + threadIdx.x;
    int src_y = blockIdx.y * BF16_TR_TILE + threadIdx.y;
    for (int j = 0; j < BF16_TR_TILE; j += blockDim.y) {
        int y = src_y + j;
        if (y < M && src_x < K) {
            tile[threadIdx.y + j][threadIdx.x] = src[y * K + src_x];
        }
    }
    __syncthreads();

    int dst_x = blockIdx.y * BF16_TR_TILE + threadIdx.x;
    int dst_y = blockIdx.x * BF16_TR_TILE + threadIdx.y;
    for (int j = 0; j < BF16_TR_TILE; j += blockDim.y) {
        int y = dst_y + j;
        if (y < K && dst_x < M) {
            dst[y * M + dst_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void bf16_transpose_into(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "bf16_transpose_into expects CUDA tensors");
    TORCH_CHECK(src.scalar_type() == torch::kBFloat16 && dst.scalar_type() == torch::kBFloat16,
                "bf16_transpose_into expects BF16 tensors");
    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "bf16_transpose_into expects 2D tensors");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(),
                "bf16_transpose_into expects contiguous tensors");
    const int64_t M = src.size(0);
    const int64_t K = src.size(1);
    TORCH_CHECK(dst.size(0) == K && dst.size(1) == M,
                "bf16_transpose_into output shape must be transpose of input");
    dim3 block(BF16_TR_TILE, 8);
    dim3 grid((K + BF16_TR_TILE - 1) / BF16_TR_TILE,
              (M + BF16_TR_TILE - 1) / BF16_TR_TILE);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bf16_transpose_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
        (int)M,
        (int)K);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bf16_transpose_into failed: ", cudaGetErrorString(err));
}

static void launch_group_quantize_v2_into_outputs(
    torch::Tensor input,
    const std::vector<int64_t>& split_sections,
    torch::Tensor wc_fp4_row,
    torch::Tensor amaxes,
    torch::Tensor sg_cat,
    torch::Tensor fwd_b_sg,
    torch::Tensor dgrad_b_sg,
    torch::Tensor d_offsets,
    const std::vector<torch::Tensor>& sc_row_list,
    const std::vector<torch::Tensor>& fp4_col_list,
    const std::vector<torch::Tensor>& sc_col_list
);

template <bool SR, bool RT>
static void launch_dim1_group_kernel_v2(
    CUtensorMap &tmap_in, CUtensorMap &tmap_out,
    nvfp4_scale_t *sc_ptr,
    int64_t rows, int64_t cols, int64_t scale_stride,
    grp_dim1_kernel::Dim1GroupArgs &kernel_args,
    cudaStream_t stream
) {
    dim3 grid(cols / grp_dim1_kernel::CHUNK_DIM_X,
             rows / grp_dim1_kernel::CHUNK_DIM_Y);

    constexpr size_t in_bytes = grp_dim1_kernel::BUFFS_NUM *
        grp_dim1_kernel::BUFF_DIM_Y * grp_dim1_kernel::BUFF_DIM_X * sizeof(__nv_bfloat16);
    constexpr size_t out_bytes = grp_dim1_kernel::BUFFS_NUM *
        grp_dim1_kernel::BUFF_DIM_Y * ((grp_dim1_kernel::BUFF_DIM_X * 4) / 8);
    constexpr size_t out_t_bytes = RT ?
        grp_dim1_kernel::BUFFS_NUM * grp_dim1_kernel::BUFF_OUT_T_SIZE : 0;
    constexpr size_t col_scales_bytes = RT ?
        grp_dim1_kernel::CHUNK_DIM_X * grp_dim1_kernel::SCALES_PER_CHUNK_Y : 0;
    constexpr size_t shmem =
        DIVUP_TO_MULTIPLE(in_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(out_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(out_t_bytes, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(col_scales_bytes, TMA_SHMEM_ALIGNMENT) +
        TMA_SHMEM_ALIGNMENT;

    auto func = grp_dim1_kernel::group_quantize_transpose_dim1_nvfp4_kernel<SR, RT>;
    // Set shmem attribute only once (first call) — skip during CUDA graph capture
    static bool s_d1_attr_set = false;
    if (!s_d1_attr_set) {
        cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        s_d1_attr_set = true;
    }
    func<<<grid, grp_dim1_kernel::THREADS_NUM, shmem, stream>>>(
        tmap_in, tmap_out, sc_ptr, nullptr, rows, cols, scale_stride,
        nullptr, kernel_args);
}

// sg compute helper
__global__ void compute_sg_kernel(const float* __restrict__ amaxes,
                                   float* __restrict__ sgs,
                                   int num) {
    int i = threadIdx.x;
    if (i < num) sgs[i] = amaxes[i] / 2688.0f;
}


// ═══════════════════════════════════════════════════════════════════
// Cached device/occupancy info (initialized once, reused every call)
// ═══════════════════════════════════════════════════════════════════
struct CachedDeviceInfo {
    int num_sms         = 0;
    int v3_max_bps      = 0;  // v3 fused max blocks/SM
    int v3_max_bps_t    = 0;  // v3 fused max blocks/SM (transpose variant)
    int v4_max_bps      = 0;  // v5 persistent max blocks/SM
    int v4_max_bps_t    = 0;  // v5 persistent max blocks/SM (transpose variant)
    int grp_d0_max_bps  = 0;  // grouped dim=0 fused
    int grp_d1_max_bps  = 0;  // grouped dim=1 fused
    int grp_d1_sr_max_bps = 0;  // grouped dim=1 fused with data SR
    int pg_d0_max_bps   = 0;  // persistent grouped dim=0
    int pg_d1_max_bps   = 0;  // persistent grouped dim=1
    int pg_d1_sr_max_bps = 0;  // persistent grouped dim=1 with data SR
    int v3_dshmem       = 0;
    int v3_dshmem_t     = 0;
    bool initialized    = false;
};

static CachedDeviceInfo& get_cached_info() {
    static CachedDeviceInfo info;
    if (!info.initialized) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&info.num_sms, cudaDevAttrMultiProcessorCount, dev);

        using namespace tk_v3;
        info.v3_dshmem   = v3_shmem_size<false>();
        info.v3_dshmem_t = v3_shmem_size<true>();

        // v3 fused kernel occupancy
        cudaFuncSetAttribute(fused_amax_quantize_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v3_max_bps, fused_amax_quantize_kernel<false>, V3_THREADS, info.v3_dshmem);

        cudaFuncSetAttribute(fused_amax_quantize_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v3_max_bps_t, fused_amax_quantize_kernel<true>, V3_THREADS, info.v3_dshmem_t);

        // v5 persistent kernel occupancy
        cudaFuncSetAttribute(tk_v5::persistent_quantize_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v4_max_bps, tk_v5::persistent_quantize_kernel<false>, V3_THREADS, info.v3_dshmem);

        cudaFuncSetAttribute(tk_v5::persistent_quantize_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.v4_max_bps_t, tk_v5::persistent_quantize_kernel<true>, V3_THREADS, info.v3_dshmem_t);

        // v2 pipelined amax
        int amax_smem = pipelined_amax::smem_size();
        cudaFuncSetAttribute(pipelined_amax::fused_amax_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);

        // Grouped dim=0 fused
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim0<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.grp_d0_max_bps, fused_group_quantize_kernel_dim0<true>, V3_THREADS, info.v3_dshmem_t);

        // Grouped dim=1 fused
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim1<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.grp_d1_max_bps, fused_group_quantize_kernel_dim1<true>, V3_THREADS, info.v3_dshmem_t);
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim1<true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim1<true, true, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(fused_group_quantize_kernel_dim1<true, false, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.grp_d1_sr_max_bps, fused_group_quantize_kernel_dim1<true, true>,
            V3_THREADS, info.v3_dshmem_t);

        // Persistent grouped dim=0
        cudaFuncSetAttribute(tk_v5::persistent_group_quantize_kernel_dim0<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim0<true, true, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim0<true, true, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.pg_d0_max_bps, tk_v5::persistent_group_quantize_kernel_dim0<true>, V3_THREADS, info.v3_dshmem_t);

        // Persistent grouped dim=1
        cudaFuncSetAttribute(tk_v5::persistent_group_quantize_kernel_dim1<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, false, true, false, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, false, true, false, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, true, true, true, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, true, true, false, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, false, true, true, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.pg_d1_max_bps, tk_v5::persistent_group_quantize_kernel_dim1<true>, V3_THREADS, info.v3_dshmem_t);
        cudaFuncSetAttribute(tk_v5::persistent_group_quantize_kernel_dim1<true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, true, true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaFuncSetAttribute(
            tk_v5::persistent_group_quantize_kernel_dim1<true, false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, info.v3_dshmem_t);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &info.pg_d1_sr_max_bps,
            tk_v5::persistent_group_quantize_kernel_dim1<true, true>,
            V3_THREADS, info.v3_dshmem_t);

        // Grouped pipelined amax
        cudaFuncSetAttribute(pipelined_amax::grouped_amax_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);
        cudaFuncSetAttribute(pipelined_amax::grouped_amax_dim1_pipelined_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, amax_smem);

        info.initialized = true;
    }
    return info;
}

// Threshold: use persistent kernel when grid >= this many tiles
static constexpr int PERSISTENT_THRESHOLD = 4096;

static inline bool disable_persistent_quant_for_conn1() {
    const char *mode = std::getenv("USE_TK_QUANT_DISABLE_PERSISTENT");
    if (mode != nullptr) {
        if (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "true") == 0 ||
            std::strcmp(mode, "TRUE") == 0 || std::strcmp(mode, "on") == 0) {
            return true;
        }
        if (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "false") == 0 ||
            std::strcmp(mode, "FALSE") == 0 || std::strcmp(mode, "off") == 0) {
            return false;
        }
    }
    const char *conn = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    return conn != nullptr && std::strcmp(conn, "1") == 0;
}

static inline bool use_barrier_free_norm_quant() {
    const char *mode = std::getenv("USE_TK_NORM_QUANT_TWO_PASS");
    if (mode != nullptr && mode[0] != '\0') {
        if (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "true") == 0 ||
            std::strcmp(mode, "TRUE") == 0 || std::strcmp(mode, "on") == 0) {
            return true;
        }
        if (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "false") == 0 ||
            std::strcmp(mode, "FALSE") == 0 || std::strcmp(mode, "off") == 0) {
            return false;
        }
        TORCH_CHECK(false,
                    "USE_TK_NORM_QUANT_TWO_PASS must be a boolean, got ", mode);
    }

    // A direct-launch software barrier is unsafe when NCCL/FSDP has resident
    // work on the device: subtracting blocks does not guarantee that every
    // barrier CTA is simultaneously resident. Keep the barrier-free producer
    // as the default and require an explicit opt-out for isolated benchmarks.
    return true;
}

static inline int persistent_launch_sms(const CachedDeviceInfo& ci) {
    // Leave launch capacity for NCCL progress while quantization overlaps FSDP prefetch.
    const char *value = std::getenv("USE_TK_QUANT_RESERVED_SMS");
    if (value == nullptr || value[0] == '\0') {
        return ci.num_sms;
    }
    char *end = nullptr;
    const long reserved = std::strtol(value, &end, 10);
    TORCH_CHECK(
        end != value && *end == '\0' && reserved >= 0 && reserved < ci.num_sms,
        "USE_TK_QUANT_RESERVED_SMS must be an integer in [0, ",
        ci.num_sms - 1, "], got ", value);
    return ci.num_sms - static_cast<int>(reserved);
}

static inline bool split3_two_pass_sr_enabled() {
    const auto is_true = [](const char *value) {
        return value != nullptr &&
               (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
                std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0);
    };
    const auto is_false = [](const char *value) {
        return value != nullptr &&
               (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
                std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "off") == 0);
    };

    // The frozen CUDA-graph ABI still launches the legacy grouped kernel.
    // Keep eager and capture warmup on the same route until that ABI is moved
    // to the barrier-free implementation as well.
    if (is_true(std::getenv("USE_CUDA_GRAPH"))) {
        return false;
    }

    const char *mode = std::getenv("USE_TK_SPLIT3_TWO_PASS");
    if (mode == nullptr || mode[0] == '\0' || is_true(mode)) {
        return true;
    }
    TORCH_CHECK(
        is_false(mode),
        "USE_TK_SPLIT3_TWO_PASS must be a boolean, got ", mode);
    return false;
}

static inline bool barrier_free_group_quant_enabled() {
    const char *mode = std::getenv("USE_TK_GROUP_QUANT_TWO_PASS");
    if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "1") == 0 ||
        std::strcmp(mode, "true") == 0 || std::strcmp(mode, "TRUE") == 0 ||
        std::strcmp(mode, "on") == 0) {
        return true;
    }
    if (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "false") == 0 ||
        std::strcmp(mode, "FALSE") == 0 || std::strcmp(mode, "off") == 0) {
        return false;
    }
    TORCH_CHECK(false,
                "USE_TK_GROUP_QUANT_TWO_PASS must be a boolean, got ", mode);
    return false;
}

static inline bool disable_fused_rht_quant() {
    const char *mode = std::getenv("NVFP4_TK_DISABLE_FUSED_RHT_QUANT");
    return mode != nullptr &&
           (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "true") == 0 ||
            std::strcmp(mode, "TRUE") == 0 || std::strcmp(mode, "on") == 0);
}

namespace tk_v5_nhsd_wo {

using namespace tk_v3;

__device__ __forceinline__
void load_nhsd_tile(
    V3_IType3D& sIn,
    uint64_t* in_mbar,
    const CUtensorMap& tensor_map_input,
    int block_offset_Y,
    int block_offset_X,
    int tile_id,
    int B,
    int H,
    int S,
    int D
) {
    (void)B;
    constexpr int shmem_tile_bytes = V3_BUFF_DIM_Y * V3_BUFF_DIM_X * sizeof(IType);
    const int stage_Y = tile_id / V3_TILES_X;
    const int stage_X = tile_id % V3_TILES_X;
    const int global_k = block_offset_X + stage_X * V3_TILE_DIM_X;
    const int h = global_k / D;
    const int d = global_k - h * D;
    const int b = block_offset_Y / S;
    const int s_base = block_offset_Y - b * S;
    const int input_row = (b * H + h) * S + s_base + stage_Y * V3_TILE_DIM_Y;
    ptx::mbarrier_arrive_expect_tx(&in_mbar[tile_id], shmem_tile_bytes);
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t*>(&sIn[tile_id]),
        reinterpret_cast<const uint64_t*>(&tensor_map_input),
        d,
        input_row,
        &in_mbar[tile_id]);
}

template <bool RETURN_TRANSPOSE, bool ENCODE_CENTRIC = true>
__global__ void __launch_bounds__(V3_THREADS)
persistent_quantize_nhsd_wo_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t,
    const __grid_constant__ CUtensorMap tmap_scale_row,
    const __grid_constant__ CUtensorMap tmap_scale_col,
    nvfp4_scale_t* const scales_ptr,
    const size_t rows,
    const size_t cols,
    const size_t scale_stride,
    int B,
    int H,
    int S,
    int D,
    tk_v5::PersistentArgs args
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    (void)scales_ptr;
    (void)scale_stride;
    const bool leading = (threadIdx.x == 0);

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_IN * V3_BUFF_IN_ELEMS * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT * V3_BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int out_tr_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3_BUFFS_NUM_OUT_TR * V3_BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
    constexpr int sc_row_bytes = DIVUP_TO_MULTIPLE(
        V3Config::CHUNK_DIM_Y * V3_SCALES_PER_CHUNK_X * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
    constexpr int sc_col_bytes = RETURN_TRANSPOSE ?
        DIVUP_TO_MULTIPLE(V3Config::CHUNK_DIM_X * V3_SCALES_PER_CHUNK_Y * (int)sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT) : 0;

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

    __shared__ uint64_t in_mbar[V3_NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    float block_max = 0.0f;
    int mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id;
        if (leading) {
            s_chunk_id = atomicAdd(args.work_counter_phase1, 1);
        }
        __syncthreads();
        if (s_chunk_id >= (unsigned int)args.total_tiles) {
            break;
        }

        const int ctaid_X = s_chunk_id % args.tiles_X;
        const int ctaid_Y = s_chunk_id / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        if (leading) {
            #pragma unroll
            for (int t = 0; t < V3_NUM_TILES; ++t) {
                load_nhsd_tile(sIn, in_mbar, tensor_map_input, block_offset_Y, block_offset_X,
                               t, B, H, S, D);
            }
        }

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
            block_max = fmaxf(block_max, scan_tile_amax(sIn_ptr, t));
        }
        mbar_phase ^= 1;
    }

    {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, mask));
        }

        __shared__ float warp_max[V3_THREADS / 32];
        int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) {
            warp_max[wid] = block_max;
        }
        __syncthreads();
        if (wid == 0) {
            block_max = (lane < V3_THREADS / 32) ? warp_max[lane] : 0.0f;
            #pragma unroll
            for (int mask = (V3_THREADS / 32) / 2; mask > 0; mask >>= 1) {
                block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, mask));
            }
        }
    }

    grid_barrier(block_max, args.global_amax,
                 args.done_counter, args.ready_flag,
                 args.num_persistent);

    const float amax_val = args.global_amax[0];
    const float S_enc = compute_global_encode_scaling_factor_FP4(amax_val);

    if (leading && blockIdx.x == 0) {
        if (args.sg_output) {
            args.sg_output[0] = amax_val / 2688.0f;
        }
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
            ptx::mbarrier_init(&in_mbar[t], 1);
        }
        ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    mbar_phase = 0;

    while (true) {
        __shared__ unsigned int s_chunk_id2;
        if (leading) {
            s_chunk_id2 = atomicAdd(args.work_counter_phase2, 1);
        }
        __syncthreads();
        if (s_chunk_id2 >= (unsigned int)args.total_tiles) {
            break;
        }

        const int ctaid_X = s_chunk_id2 % args.tiles_X;
        const int ctaid_Y = s_chunk_id2 / args.tiles_X;
        const int block_offset_Y = ctaid_Y * V3Config::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * V3Config::CHUNK_DIM_X;

        if (leading) {
            #pragma unroll
            for (int t = 0; t < V3_NUM_TILES; ++t) {
                load_nhsd_tile(sIn, in_mbar, tensor_map_input, block_offset_Y, block_offset_X,
                               t, B, H, S, D);
            }
        }

        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);
        }

        tk_v5::quantize_and_store_chunk_v5<RETURN_TRANSPOSE, ENCODE_CENTRIC>(
            sIn_ptr, sOut_ptr, sOut_tr_ptr,
            sSFrowwise_ptr, sSFcolwise_ptr,
            sOut, sOut_tr, sSFrowwise, sSFcolwise,
            tensor_map_output, tensor_map_output_t,
            tmap_scale_row, tmap_scale_col,
            S_enc,
            block_offset_Y, block_offset_X,
            (int)rows, (int)cols,
            ctaid_X, ctaid_Y);
        mbar_phase ^= 1;
    }

    if (leading) {
        #pragma unroll
        for (int t = 0; t < V3_NUM_TILES; ++t) {
            ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
#else
    NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

}  // namespace tk_v5_nhsd_wo

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_impl(
    torch::Tensor input,
    int64_t output_rows,
    int64_t output_cols,
    bool return_transpose,
    bool encode_centric
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t input_rows = input.size(0);
    const int64_t input_cols = input.size(1);
    const int64_t M = output_rows;
    const int64_t K = output_cols;
    TORCH_CHECK(input_rows % 128 == 0 && input_cols % 128 == 0);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);
    TORCH_CHECK(M >= input_rows && K >= input_cols,
                "padded quantized extent must cover the logical input");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    V5QuantSequenceGuard sequence_guard(stream);
    auto device = input.device();

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4) : torch::empty({0}, opts_fp4);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8) : torch::empty({0}, opts_fp8);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto amax_buf = torch::empty({2}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;

    auto sync_buf = torch::empty({4}, opts_u32);
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    // Grid dimensions
    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached_info();
    const int v3_dshmem = return_transpose ? ci.v3_dshmem_t : ci.v3_dshmem;

    if (total_tiles < PERSISTENT_THRESHOLD || disable_persistent_quant_for_conn1()) {
        // ─── v3 path (fused single-pass if grid fits, v2 fallback otherwise) ───
        const int max_bps = return_transpose ? ci.v3_max_bps_t : ci.v3_max_bps;
        const int max_concurrent = max_bps * persistent_launch_sms(ci);
        const bool can_fuse = (total_tiles <= max_concurrent && max_bps > 0);

        if (can_fuse) {
            // v3 fused single-pass
            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in, input.data_ptr(), input_rows, input_cols,
                          V3_BUFF_DIM_Y, V3_BUFF_DIM_X, input_cols, 16);
            create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            unsigned int *done = sync_data + 2, *ready = sync_data + 3;
            if (encode_centric) {
                if (return_transpose)
                    launch_v3<true, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v3<true, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            }
        } else {
            // v2 two-pass fallback
            int64_t n = input_rows * input_cols;
            int amax_blocks = pipelined_amax::grid_size(n);
            int amax_smem = pipelined_amax::smem_size();
            pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                amax_ptr, sg_ptr, sync_data, n);

            alignas(64) CUtensorMap ti{}, to{}, tot{};
            create_tma_2d(ti, input.data_ptr(), input_rows, input_cols,
                          BUFF_DIM_Y, BUFF_DIM_X, input_cols, 16);
            create_tma_2d(to, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tot, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

            if (encode_centric) {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            }
        }
    }
    else {
        // ─── v5 persistent (work-stealing, L2 promotion, TMA scale output) ───
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), input_rows, input_cols,
                      V3_BUFF_DIM_Y, V3_BUFF_DIM_X, input_cols, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        // TMA maps for scale tensors: view [ntm, ntk, 512] as 2D using BF16 type
        // Max TMA box = 512 bytes, so use shmemX=256 BF16 (512 bytes) per tile_k block
        // Each chunk does 2 TMA stores (one per tile_k × 512-byte block)
        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;  // ntk_r*512 bytes / 2 = BF16 elements
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        if (return_transpose && sc_t_ptr) {
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }

        const int p_max_bps = return_transpose ? ci.v4_max_bps_t : ci.v4_max_bps;
        int num_persistent = p_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;
        pargs.col_scales_ptr = sc_t_ptr;
        pargs.col_scale_stride = scale_stride_t;
        pargs.swizzle_scales = true;

        cudaError_t launch_err = cudaSuccess;
        if (encode_centric) {
            if (return_transpose)
                launch_err = launch_persistent_quantize_stream_safe<true, true>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs,
                    num_persistent, v3_dshmem, stream);
            else
                launch_err = launch_persistent_quantize_stream_safe<false, true>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs,
                    num_persistent, v3_dshmem, stream);
        } else {
            if (return_transpose)
                launch_err = launch_persistent_quantize_stream_safe<true, false>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs,
                    num_persistent, v3_dshmem, stream);
            else
                launch_err = launch_persistent_quantize_stream_safe<false, false>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs,
                    num_persistent, v3_dshmem, stream);
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "stream-safe persistent quantize launch failed: ",
                    cudaGetErrorString(launch_err));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v5_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1),
                           amax_buf, sync_buf);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm(torch::Tensor input, bool return_transpose, bool encode_centric) {
    return tk_v4_quantize_for_gemm_impl(
        input, input.size(0), input.size(1), return_transpose, encode_centric);
}

__global__ void swizzle_nvfp4_2d_weight_scales_kernel(
    const uint32_t* __restrict__ row_source,
    const uint32_t* __restrict__ col_source,
    uint32_t* __restrict__ row_output,
    uint32_t* __restrict__ col_output,
    int64_t quartets,
    int64_t row_ntk,
    int64_t col_ntk
) {
    const int64_t quartet_offset =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (quartet_offset >= quartets) return;

    const int64_t tile = quartet_offset / 128;
    const int64_t quartet = quartet_offset % 128;
    const int64_t source_row = (quartet % 4) * 32 + quartet / 4;

    const int64_t row_tm = tile / row_ntk;
    const int64_t row_tk = tile % row_ntk;
    const int64_t row_source_offset =
        (row_tm * 128 + source_row) * row_ntk + row_tk;

    const int64_t col_tm = tile / col_ntk;
    const int64_t col_tk = tile % col_ntk;
    const int64_t col_source_offset =
        (col_tm * 128 + source_row) * col_ntk + col_tk;

    row_output[quartet_offset] = row_source[row_source_offset];
    col_output[quartet_offset] = col_source[col_source_offset];
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_quantize_weight_2d(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be bf16 [M, K]");
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "2D v5 weights require M and K to be multiples of 128");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
    // This producer is intentionally overlapped with activation quantization.
    // Its amax completion state is operation-local, so joining the global v5
    // stream chain would add a second ordering graph and can create a cycle.
    auto device = input.device();
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto row_sc_raw = torch::empty({M, K / 16}, opts_u8);
    auto col_sc_raw = torch::empty({K, M / 16}, opts_u8);
    auto row_sc = torch::empty({M / 128, K / 64, 512}, opts_fp8);
    auto col_sc = torch::empty({K / 128, M / 64, 512}, opts_fp8);
    auto amax_buf = torch::empty({3}, opts_f32);
    float* amax_ptr = amax_buf.data_ptr<float>();
    float* sg_ptr = amax_ptr + 1;
    auto* amax_block_count = reinterpret_cast<unsigned int*>(amax_ptr + 2);

    cudaMemsetAsync(amax_ptr, 0, 3 * sizeof(float), stream);
    const int64_t elements = M * K;
    const int amax_blocks = pipelined_amax::grid_size(elements);
    const int amax_smem = pipelined_amax::smem_size();
    pipelined_amax::fused_amax_pipelined_kernel<<<
        amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        amax_ptr,
        sg_ptr,
        amax_block_count,
        elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    namespace q2d =
        transformer_engine::dispatch::nvfp4::quantize_transpose_kernel;
    using IType = transformer_engine::bf16;
    using ParamOP = transformer_engine::Empty;
    constexpr bool COMPUTE_ACTIVATIONS = false;
    constexpr float (*OP)(float, const ParamOP&) = nullptr;
    constexpr bool USE_STOCHASTIC_ROUNDING = false;
    constexpr bool RETURN_TRANSPOSE = true;
    // TE's 2D weight path currently dispatches the decode-centric kernel.
    constexpr bool ENCODE_CENTRIC = false;

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K,
                  q2d::BUFF_DIM_Y, q2d::BUFF_DIM_X, K, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K,
                  q2d::BUFF_DIM_Y, q2d::BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M,
                  q2d::BUFF_DIM_X, q2d::BUFF_DIM_Y, M, 4);

    constexpr size_t buff_elems = q2d::BUFF_DIM_Y * q2d::BUFF_DIM_X;
    constexpr size_t buff_elems_total = q2d::BUFFS_NUM * buff_elems;
    constexpr size_t in_mem = DIVUP_TO_MULTIPLE(
        buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr size_t out_mem = 2 * DIVUP_TO_MULTIPLE(
        (buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
    constexpr size_t scale_mem =
        (q2d::CHUNK_DIM_Y * q2d::CHUNK_DIM_X) / 16 *
        sizeof(nvfp4_scale_t);
    constexpr size_t dshmem =
        in_mem + out_mem + scale_mem + TMA_SHMEM_ALIGNMENT;

    auto kernel = q2d::quantize_transpose_nvfp4_2D_kernel<
        COMPUTE_ACTIVATIONS, ParamOP, OP, IType,
        USE_STOCHASTIC_ROUNDING, RETURN_TRANSPOSE, ENCODE_CENTRIC>;
    v5_check_cuda(
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem),
        "2D v5 weight quantizer shared-memory setup failed");
    const dim3 grid(K / q2d::CHUNK_DIM_X, M / q2d::CHUNK_DIM_Y);
    kernel<<<grid, q2d::THREADS_NUM, dshmem, stream>>>(
        tmap_in, tmap_out, tmap_out_t,
        reinterpret_cast<nvfp4_scale_t*>(row_sc_raw.data_ptr()),
        reinterpret_cast<nvfp4_scale_t*>(col_sc_raw.data_ptr()),
        nullptr, amax_ptr, amax_ptr,
        M, K, K / 16, M / 16, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int64_t quartets = row_sc.numel() / 4;
    constexpr int threads = 256;
    swizzle_nvfp4_2d_weight_scales_kernel<<<
        (quartets + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(row_sc_raw.data_ptr()),
        reinterpret_cast<const uint32_t*>(col_sc_raw.data_ptr()),
        reinterpret_cast<uint32_t*>(row_sc.data_ptr()),
        reinterpret_cast<uint32_t*>(col_sc.data_ptr()),
        quartets, K / 64, M / 64);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        row_fp4, row_sc, col_fp4, col_sc,
        amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1));
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor>
tk_gated_group_rmsnorm_quantize_for_gemm(
    torch::Tensor scan,
    torch::Tensor gate,
    torch::Tensor gamma,
    double epsilon,
    bool encode_centric
) {
    namespace gated_quant = gated_group_rmsnorm_quant;
    constexpr int64_t kHidden = gated_quant::kHidden;
    constexpr int64_t kGateRowStride = 18688;

    TORCH_CHECK(
        scan.is_cuda() && scan.scalar_type() == torch::kBFloat16 &&
            scan.dim() == 2 && scan.is_contiguous(),
        "scan must be contiguous CUDA BF16 [M,8192]");
    TORCH_CHECK(
        gate.is_cuda() && gate.scalar_type() == torch::kBFloat16 &&
            gate.dim() == 2 && gate.size(1) == kHidden &&
            gate.stride(1) == 1 && gate.stride(0) == kGateRowStride,
        "gate must be CUDA BF16 [M,8192] with row stride 18688");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.scalar_type() == torch::kBFloat16 &&
            gamma.dim() == 1 && gamma.is_contiguous() &&
            gamma.numel() == kHidden,
        "gamma must be contiguous CUDA BF16 [8192]");
    TORCH_CHECK(
        scan.device() == gate.device() && scan.device() == gamma.device(),
        "scan, gate, and gamma must be on one CUDA device");
    TORCH_CHECK(
        scan.size(1) == kHidden && gate.size(0) == scan.size(0),
        "scan and gate must have matching [M,8192] logical shapes");
    TORCH_CHECK(
        scan.size(0) == 8192 || scan.size(0) == 16384 ||
            scan.size(0) == 24576 || scan.size(0) == 32768,
        "M must be one of {8192,16384,24576,32768}");
    TORCH_CHECK(
        std::isfinite(epsilon) && epsilon >= 0.0,
        "epsilon must be finite and non-negative");

    const int64_t M = scan.size(0);
    const int64_t K = scan.size(1);
    const c10::cuda::CUDAGuard device_guard(scan.device());
    auto stream =
        at::cuda::getCurrentCUDAStream(scan.get_device()).stream();
    V5QuantSequenceGuard sequence_guard(stream);

    auto fp4_options = torch::dtype(torch::kFloat4_e2m1fn_x2)
                           .device(scan.device());
    auto fp8_options = torch::dtype(torch::kFloat8_e4m3fn)
                           .device(scan.device());
    auto f32_options =
        torch::dtype(torch::kFloat32).device(scan.device());
    auto i32_options =
        torch::dtype(torch::kInt32).device(scan.device());

    auto normalized = torch::empty({M, K}, scan.options());
    const int64_t row_scale_m = M / 128;
    const int64_t row_scale_k = K / 64;
    const int64_t col_scale_m = K / 128;
    const int64_t col_scale_k = M / 64;
    auto row_fp4 = torch::empty({M, K / 2}, fp4_options);
    auto row_scales = torch::empty(
        {row_scale_m, row_scale_k, 512}, fp8_options);
    auto col_fp4 = torch::empty({K, M / 2}, fp4_options);
    auto col_scales = torch::empty(
        {col_scale_m, col_scale_k, 512}, fp8_options);
    auto amax_buffer = torch::empty({2}, f32_options);
    auto sync_buffer = torch::empty({4}, i32_options);
    auto inv_rms = torch::empty(
        {M, gated_quant::kGroupsPerRow}, f32_options);

    float* amax_ptr = amax_buffer.data_ptr<float>();
    auto* sync_ptr = reinterpret_cast<unsigned int*>(
        sync_buffer.data_ptr<int32_t>());
    v5_check_cuda(
        cudaMemsetAsync(
            amax_ptr, 0, 2 * sizeof(float), stream),
        "gated group RMSNorm amax reset");
    v5_check_cuda(
        cudaMemsetAsync(
            sync_ptr, 0, 4 * sizeof(unsigned int), stream),
        "gated group RMSNorm sync reset");

    auto stats_kernel = gated_quant::gated_group_rmsnorm_amax_kernel;
    static int stats_blocks_per_sm = -1;
    if (stats_blocks_per_sm < 0) {
        v5_check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &stats_blocks_per_sm,
                stats_kernel,
                gated_quant::kThreads,
                0),
            "gated group RMSNorm stats occupancy query");
    }
    const int total_groups =
        static_cast<int>(M) * gated_quant::kGroupsPerRow;
    const int stats_blocks = std::min(
        total_groups,
        stats_blocks_per_sm * get_cached_info().num_sms);
    TORCH_CHECK(
        stats_blocks > 0,
        "gated group RMSNorm stats kernel has no resident CTA");
    stats_kernel<<<
        stats_blocks,
        gated_quant::kThreads,
        0,
        stream>>>(
        reinterpret_cast<const tk_v3::IType*>(scan.data_ptr()),
        reinterpret_cast<const tk_v3::IType*>(gate.data_ptr()),
        reinterpret_cast<const tk_v3::IType*>(gamma.data_ptr()),
        reinterpret_cast<tk_v3::IType*>(normalized.data_ptr()),
        inv_rms.data_ptr<float>(),
        amax_ptr,
        static_cast<int>(M),
        gate.stride(0),
        static_cast<float>(epsilon));
    v5_check_cuda(
        cudaGetLastError(),
        "gated group RMSNorm stats/amax launch");

    using namespace tk_v3;
    const int tiles_x = K / V3Config::CHUNK_DIM_X;
    const int tiles_y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_x * tiles_y;
    const int64_t scale_stride =
        ((K / V3_SCALE_DIM) + 3) / 4 * 4;

    alignas(64) CUtensorMap input_map{};
    alignas(64) CUtensorMap row_output_map{};
    alignas(64) CUtensorMap col_output_map{};
    alignas(64) CUtensorMap row_scale_map{};
    alignas(64) CUtensorMap col_scale_map{};
    create_tma_2d(
        input_map,
        normalized.data_ptr(),
        M,
        K,
        V3_BUFF_DIM_Y,
        V3_BUFF_DIM_X,
        K,
        16,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(
        row_output_map,
        row_fp4.data_ptr(),
        M,
        K,
        V3_BUFF_DIM_Y,
        V3_BUFF_DIM_X,
        K,
        4);
    create_tma_2d(
        col_output_map,
        col_fp4.data_ptr(),
        K,
        M,
        V3_BUFF_DIM_X,
        V3_BUFF_DIM_Y,
        M,
        4);
    const int64_t row_scale_x = row_scale_k * 256;
    const int64_t col_scale_x = col_scale_k * 256;
    create_tma_2d(
        row_scale_map,
        row_scales.data_ptr(),
        row_scale_m,
        row_scale_x,
        1,
        256,
        row_scale_x,
        16);
    create_tma_2d(
        col_scale_map,
        col_scales.data_ptr(),
        col_scale_m,
        col_scale_x,
        1,
        256,
        col_scale_x,
        16);

    auto encode_phase2_kernel =
        tk_v5::persistent_quantize_phase2_kernel<true, true>;
    auto decode_phase2_kernel =
        tk_v5::persistent_quantize_phase2_kernel<true, false>;
    static int phase2_dynamic_smem = -1;
    static int encode_phase2_blocks_per_sm = -1;
    static int decode_phase2_blocks_per_sm = -1;
    if (phase2_dynamic_smem < 0) {
        phase2_dynamic_smem = get_cached_info().v3_dshmem_t;
        v5_check_cuda(
            cudaFuncSetAttribute(
                encode_phase2_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                phase2_dynamic_smem),
            "gated group RMSNorm encode producer shared-memory setup");
        v5_check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &encode_phase2_blocks_per_sm,
                encode_phase2_kernel,
                V3_THREADS,
                phase2_dynamic_smem),
            "gated group RMSNorm encode producer occupancy query");
        v5_check_cuda(
            cudaFuncSetAttribute(
                decode_phase2_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                phase2_dynamic_smem),
            "gated group RMSNorm decode producer shared-memory setup");
        v5_check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &decode_phase2_blocks_per_sm,
                decode_phase2_kernel,
                V3_THREADS,
                phase2_dynamic_smem),
            "gated group RMSNorm decode producer occupancy query");
    }
    const int phase2_blocks_per_sm = encode_centric
        ? encode_phase2_blocks_per_sm
        : decode_phase2_blocks_per_sm;
    const int phase2_blocks = std::min(
        total_tiles,
        phase2_blocks_per_sm * get_cached_info().num_sms);
    TORCH_CHECK(
        phase2_blocks > 0,
        "gated group RMSNorm producer has no resident CTA");

    tk_v5::Phase2Args phase2_args{};
    phase2_args.work_counter = sync_ptr;
    phase2_args.global_amax = amax_ptr;
    phase2_args.tiles_X = tiles_x;
    phase2_args.tiles_Y = tiles_y;
    phase2_args.total_tiles = total_tiles;
    phase2_args.sg_output = amax_ptr + 1;
    auto phase2_kernel = encode_centric
        ? encode_phase2_kernel
        : decode_phase2_kernel;
    phase2_kernel<<<
        phase2_blocks,
        V3_THREADS,
        phase2_dynamic_smem,
        stream>>>(
        input_map,
        row_output_map,
        col_output_map,
        row_scale_map,
        col_scale_map,
        reinterpret_cast<nvfp4_scale_t*>(row_scales.data_ptr()),
        M,
        K,
        scale_stride,
        phase2_args);
    v5_check_cuda(
        cudaGetLastError(),
        "gated group RMSNorm producer launch");

    auto global_scale = amax_buffer.narrow(0, 1, 1);
    return std::make_tuple(
        row_fp4,
        row_scales,
        col_fp4,
        col_scales,
        global_scale,
        global_scale,
        amax_buffer,
        sync_buffer,
        inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_padded(
    torch::Tensor input,
    int64_t output_rows,
    int64_t output_cols,
    bool return_transpose,
    bool encode_centric
) {
    TORCH_CHECK(output_rows % 256 == 0 && output_cols % 256 == 0,
                "TK GEMM padded quantization requires 256-aligned output dimensions");
    return tk_v4_quantize_for_gemm_impl(
        input, output_rows, output_cols, return_transpose, encode_centric);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_delayed(
    torch::Tensor input,
    torch::Tensor prev_amax,
    bool return_transpose,
    bool encode_centric,
    bool collect_current_amax
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(prev_amax.is_cuda() && prev_amax.is_contiguous());
    TORCH_CHECK(prev_amax.scalar_type() == torch::kFloat32 && prev_amax.numel() >= 1);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    V5QuantSequenceGuard sequence_guard(stream);
    auto device = input.device();

    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4) : torch::empty({0}, opts_fp4);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8) : torch::empty({0}, opts_fp8);
    auto amax_buf = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({1}, opts_i32);
    float* cur_amax = amax_buf.data_ptr<float>();
    float* sg_ptr = cur_amax + 1;
    auto* sync = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(sync, 0, sizeof(unsigned int), stream);
    if (collect_current_amax) {
        cudaMemsetAsync(cur_amax, 0, sizeof(float), stream);
        int64_t n = M * K;
        int amax_blocks = pipelined_amax::grid_size(n);
        int amax_smem = pipelined_amax::smem_size();
        pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            cur_amax, sg_ptr, sync, n);
    }

    const auto& ci = get_cached_info();
    static int s_delay_p2_dshmem = -1;
    static int s_delay_p2_max_bps = -1;
    static int s_delay_p2_t_dshmem = -1;
    static int s_delay_p2_t_max_bps = -1;
    if (s_delay_p2_dshmem < 0) {
        s_delay_p2_dshmem = ci.v3_dshmem;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_delay_p2_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_delay_p2_max_bps,
            tk_v5::persistent_quantize_phase2_kernel<false>,
            V3_THREADS, s_delay_p2_dshmem);

        s_delay_p2_t_dshmem = ci.v3_dshmem_t;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_delay_p2_t_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_delay_p2_t_max_bps,
            tk_v5::persistent_quantize_phase2_kernel<true>,
            V3_THREADS, s_delay_p2_t_dshmem);
    }

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    if (return_transpose) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
    }

    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    if (return_transpose) {
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    tk_v5::Phase2Args args;
    memset(&args, 0, sizeof(args));
    args.work_counter = sync;
    args.global_amax = prev_amax.data_ptr<float>();
    args.tiles_X = tiles_X;
    args.tiles_Y = tiles_Y;
    args.total_tiles = total_tiles;
    args.sg_output = sg_ptr;

    const int max_bps = return_transpose ? s_delay_p2_t_max_bps : s_delay_p2_max_bps;
    int num_persistent = max_bps * persistent_launch_sms(ci);
    if (num_persistent > total_tiles) num_persistent = total_tiles;
    const dim3 grid(num_persistent);

    if (encode_centric) {
        if (return_transpose) {
            tk_v5::persistent_quantize_phase2_kernel<true, true>
                <<<grid, V3_THREADS, s_delay_p2_t_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t,
                    tmap_sc_row, tmap_sc_col,
                    reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()),
                    M, K, scale_stride, args);
        } else {
            tk_v5::persistent_quantize_phase2_kernel<false, true>
                <<<grid, V3_THREADS, s_delay_p2_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t,
                    tmap_sc_row, tmap_sc_col,
                    reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()),
                    M, K, scale_stride, args);
        }
    } else {
        if (return_transpose) {
            tk_v5::persistent_quantize_phase2_kernel<true, false>
                <<<grid, V3_THREADS, s_delay_p2_t_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t,
                    tmap_sc_row, tmap_sc_col,
                    reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()),
                    M, K, scale_stride, args);
        } else {
            tk_v5::persistent_quantize_phase2_kernel<false, false>
                <<<grid, V3_THREADS, s_delay_p2_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t,
                    tmap_sc_row, tmap_sc_col,
                    reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()),
                    M, K, scale_stride, args);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v5_quantize_for_gemm_delayed failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1),
                           amax_buf, sync_buf);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_quantize_nhsd_wo_for_gemm(torch::Tensor input, bool return_transpose, bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4,
                "input must be bf16 [B, H, S, D]");

    const int64_t B = input.size(0);
    const int64_t H = input.size(1);
    const int64_t S = input.size(2);
    const int64_t D = input.size(3);
    const int64_t M = B * S;
    const int64_t K = H * D;
    TORCH_CHECK(D == 64,
                "NHSD WO direct quantization currently requires head_dim == 64");
    TORCH_CHECK(S % 128 == 0,
                "NHSD WO direct quantization requires sequence length to be a multiple of 128");
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "logical [B*S, H*D] dimensions must be multiples of 128");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4) : torch::empty({0}, opts_fp4);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8) : torch::empty({0}, opts_fp8);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto amax_buf = torch::empty({2}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;

    auto sync_buf = torch::empty({4}, opts_u32);
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in, input.data_ptr(), B * H * S, D,
                  V3_BUFF_DIM_Y, V3_BUFF_DIM_X, D, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    if (return_transpose) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
    }

    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

    if (return_transpose && sc_t_ptr) {
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    const auto& ci = get_cached_info();
    const int dshmem = return_transpose ? ci.v3_dshmem_t : ci.v3_dshmem;

    auto launch = [&](auto kernel) {
        int max_bps = 0;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_bps, kernel, V3_THREADS, dshmem);
        int num_persistent = max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;
        pargs.col_scales_ptr = sc_t_ptr;
        pargs.col_scale_stride = scale_stride_t;
        pargs.swizzle_scales = true;

        kernel<<<dim3(num_persistent), V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            tmap_sc_row, tmap_sc_col,
            sc_ptr,
            M, K, scale_stride,
            (int)B, (int)H, (int)S, (int)D,
            pargs);
    };

    if (encode_centric) {
        if (return_transpose) {
            launch(tk_v5_nhsd_wo::persistent_quantize_nhsd_wo_kernel<true, true>);
        } else {
            launch(tk_v5_nhsd_wo::persistent_quantize_nhsd_wo_kernel<false, true>);
        }
    } else {
        if (return_transpose) {
            launch(tk_v5_nhsd_wo::persistent_quantize_nhsd_wo_kernel<true, false>);
        } else {
            launch(tk_v5_nhsd_wo::persistent_quantize_nhsd_wo_kernel<false, false>);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_nhsd_wo_for_gemm failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1),
                           amax_buf, sync_buf);
}


// ═══════════════════════════════════════════════════════════════════
// 1cs. Unit-Bound quantize: Phase-2 only with fixed absmax=1.0
//
// Skips the Phase 1 global amax scan by using the analytical absmax bound
// for probabilities and cross-entropy gradients. This gives
// S_enc = 2688 and sg = 1/2688 while preserving every value in [-1, 1].
//
// Saves ~30% of per-quant time (no amax scan, no grid barrier, no sync).
// This entrypoint is valid only for tensors whose absolute values are <= 1.
//
// Returns same tuple as tk_v4_quantize_for_gemm:
//   (row_fp4, row_sc, col_fp4, col_sc, sg, sg)
// where sg is always 1/2688.
// ═══════════════════════════════════════════════════════════════════

// Cached constant amax tensor (initialized once, reused across calls)
static torch::Tensor s_const_amax;

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_quantize_for_gemm_constant_scale(torch::Tensor input, bool return_transpose) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_fp4);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_fp4) : torch::empty({0}, opts_fp4);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_fp8) : torch::empty({0}, opts_fp8);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    // P and G are bounded by one. Pinning their true analytical bound avoids
    // the amax scan without discarding 2688x of encode range.
    if (!s_const_amax.defined() || s_const_amax.device() != device) {
        s_const_amax = torch::full({2}, 1.0f, opts_f32);
        s_const_amax[1] = 1.0f / 2688.0f;
    }
    float *amax_ptr = s_const_amax.data_ptr<float>();
    float *sg_ptr   = amax_ptr + 1;

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached_info();
    const int v3_dshmem = return_transpose ? ci.v3_dshmem_t : ci.v3_dshmem;

    if (total_tiles < PERSISTENT_THRESHOLD || disable_persistent_quant_for_conn1()) {
        // ─── Small grid: use v2 quantize-only (pre-set amax, no Phase 1) ───
        // The v2 kernel reads amax from device memory and quantizes directly.
        alignas(64) CUtensorMap ti{}, to{}, tot{};
        create_tma_2d(ti, input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
        create_tma_2d(to, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tot, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

        if (return_transpose)
            launch_v2_kernel<false, false, true, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
        else
            launch_v2_kernel<false, false, false, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
    }
    else {
        // ─── Large grid: Phase-2-only persistent kernel (skip Phase 1 amax scan) ───
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        if (return_transpose && sc_t_ptr) {
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }

        const int p_max_bps = return_transpose ? ci.v4_max_bps_t : ci.v4_max_bps;
        int num_persistent = p_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        // Phase-2-only: work counter needs zeroing (one u32 instead of 4)
        auto sync_buf = torch::zeros({1}, opts_u32);
        unsigned int *work_counter = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());

        tk_v5::Phase2Args p2args;
        memset(&p2args, 0, sizeof(p2args));
        p2args.work_counter = work_counter;
        p2args.global_amax  = amax_ptr;  // analytical absmax bound 1.0f
        p2args.tiles_X = tiles_X;
        p2args.tiles_Y = tiles_Y;
        p2args.total_tiles = total_tiles;
        p2args.sg_output = sg_ptr;  // will write 1/2688

        const dim3 grid(num_persistent);
        if (return_transpose)
            tk_v5::persistent_quantize_phase2_kernel<true, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc_ptr, M, K, scale_stride, p2args);
        else
            tk_v5::persistent_quantize_phase2_kernel<false, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                sc_ptr, M, K, scale_stride, p2args);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_for_gemm_constant_scale failed: ", cudaGetErrorString(err));

    // sg is always 1/2688 for the unit-bound path.
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           s_const_amax.narrow(0, 1, 1), s_const_amax.narrow(0, 1, 1));
}


// ═══════════════════════════════════════════════════════════════════
// 1a. Col-only quantize: takes pre-computed sg, only produces col output
//
// Use after a row-only quant to produce col data for wgrad on a side stream.
// Reads sg from DEVICE memory (no D2H sync) → safe for multi-stream overlap.
// ═══════════════════════════════════════════════════════════════════

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

__device__ __forceinline__ uint32_t mxfp4_mixed_next_rbits(
    uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

template <bool DATA_SR>
__global__ void mxfp4_row_native_sg_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ row_fp4,
    uint8_t* __restrict__ row_sc,
    float* __restrict__ native_sg,
    int rows,
    int cols,
    int blocks_x,
    int total_tiles,
    const uint64_t* __restrict__ rng_state
) {
    constexpr int kTile = 128;
    constexpr int kBlock = 32;
    constexpr int kBlocksPerTile = kTile * (kTile / kBlock);
    const int tile_id = blockIdx.x;
    if (tile_id >= total_tiles) {
        return;
    }

    const int tile_x = tile_id % blocks_x;
    const int tile_y = tile_id / blocks_x;
    const int row_base = tile_y * kTile;
    const int col_base = tile_x * kTile;
    float thread_max = 0.0f;
    uint64_t rng_seed = 0;
    uint64_t rng_subsequence_base = 0;
    if constexpr (DATA_SR) {
        rng_seed = rng_state[0];
        rng_subsequence_base = rng_state[1];
    }

    for (int block = threadIdx.x; block < kBlocksPerTile; block += blockDim.x) {
        const int local_row = block / 4;
        const int block_k = block & 3;
        const int local_col = block_k * kBlock;
        const auto* input_ptr =
            input + static_cast<int64_t>(row_base + local_row) * cols +
            col_base + local_col;
        uint4 packed_input[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed_input[i] = reinterpret_cast<const uint4*>(input_ptr)[i];
        }
        const auto* values = reinterpret_cast<const __nv_bfloat16*>(packed_input);

        float block_amax = 0.0f;
#pragma unroll
        for (int i = 0; i < kBlock; ++i) {
            block_amax = fmaxf(
                block_amax, fabsf(__bfloat162float(values[i])));
        }
        thread_max = fmaxf(thread_max, block_amax);

        const uint8_t scale = mxfp4_encode_ceil_amax(block_amax);
        const float coefficient = 6.0f * mxfp4_reciprocal_e8m0(scale);
        uint4 packed_output{};
        auto* output_bytes = reinterpret_cast<uint8_t*>(&packed_output);
        if constexpr (DATA_SR) {
            const uint64_t sequence = rng_subsequence_base +
                static_cast<uint64_t>(tile_id) * kBlocksPerTile + block;
            uint32_t sr_state = tk_splitmix32(rng_seed ^ sequence);
            if (sr_state == 0) sr_state = 0x9e3779b9u;
            const float2 conversion_scale = {coefficient, coefficient};
#pragma unroll
            for (int i = 0; i < kBlock; i += 4) {
                const float2 in01 = {
                    __bfloat162float(values[i]),
                    __bfloat162float(values[i + 1]),
                };
                const float2 in23 = {
                    __bfloat162float(values[i + 2]),
                    __bfloat162float(values[i + 3]),
                };
                const auto packed4 =
                    transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<true>(
                        in01,
                        in23,
                        conversion_scale,
                        mxfp4_mixed_next_rbits(sr_state));
                const uint16_t raw =
                    *reinterpret_cast<const uint16_t*>(&packed4);
                output_bytes[i / 2] = static_cast<uint8_t>(raw & 0xffu);
                output_bytes[i / 2 + 1] =
                    static_cast<uint8_t>((raw >> 8) & 0xffu);
            }
        } else {
#pragma unroll
            for (int i = 0; i < kBlock / 2; ++i) {
                const float2 pair = make_float2(
                    __bfloat162float(values[2 * i]) * coefficient,
                    __bfloat162float(values[2 * i + 1]) * coefficient);
                output_bytes[i] = static_cast<uint8_t>(
                    __nv_cvt_float2_to_fp4x2(
                        pair, __NV_E2M1, cudaRoundNearest));
            }
        }
        auto* output_ptr =
            row_fp4 + static_cast<int64_t>(row_base + local_row) * (cols / 2) +
            (col_base + local_col) / 2;
        *reinterpret_cast<uint4*>(output_ptr) = packed_output;

        const int scale_offset =
            tile_id * 512 +
            (local_row % 32) * 16 +
            (local_row / 32) * 4 + block_k;
        row_sc[scale_offset] = scale;
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        thread_max = fmaxf(
            thread_max,
            __shfl_xor_sync(0xffffffff, thread_max, mask));
    }
    __shared__ float warp_max[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[warp] = thread_max;
    }
    __syncthreads();

    if (warp == 0) {
        const int warp_count = blockDim.x / 32;
        float tile_max = lane < warp_count ? warp_max[lane] : 0.0f;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            tile_max = fmaxf(
                tile_max,
                __shfl_xor_sync(0xffffffff, tile_max, mask));
        }
        if (lane == 0 && tile_max > 0.0f) {
            transformer_engine::atomicMaxFloat(native_sg, tile_max / 2688.0f);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_col_only(torch::Tensor input, torch::Tensor sg_tensor) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(sg_tensor.is_cuda() && sg_tensor.numel() >= 1);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();

    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    auto col_fp4 = torch::empty({K, M / 2}, opts_fp4);
    auto col_sc  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

    float *sg_device_ptr = sg_tensor.data_ptr<float>();

    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached_info();

    if (total_tiles < PERSISTENT_THRESHOLD || disable_persistent_quant_for_conn1()) {
        auto amax_tensor = sg_tensor * 2688.0f;
        auto full_result = tk_quantize_transpose(input, amax_tensor, amax_tensor, true, false);
        auto col_fp4_v = std::get<2>(full_result).view(torch::kFloat4_e2m1fn_x2);
        auto col_sc_v  = std::get<3>(full_result).view(torch::kFloat8_e4m3fn);
        auto sg_v      = sg_tensor;

        return std::make_tuple(col_fp4_v, col_sc_v, sg_v, sg_v);
    } else {
        // Large grid: use col-only persistent Phase2 kernel (no row output at all)
        alignas(64) CUtensorMap tmap_in{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_col{};
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        // Query max occupancy for col-only kernel
        static int s_col_only_dshmem = -1;
        static int s_col_only_max_bps = -1;
        if (s_col_only_dshmem < 0) {
            s_col_only_dshmem = tk_v5_col_only::col_only_shmem_size();
            auto kern = tk_v5_col_only::persistent_quantize_phase2_col_only_kernel<true>;
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, s_col_only_dshmem);
            int max_bps_val = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_bps_val, kern, V3_THREADS, s_col_only_dshmem);
            s_col_only_max_bps = max_bps_val;
        }

        int num_persistent = s_col_only_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        // Work counter: must be zeroed
        auto sync_buf = torch::zeros({1}, opts_u32);
        unsigned int *work_counter = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());

        tk_v5_col_only::Phase2ColOnlyArgs args;
        memset(&args, 0, sizeof(args));
        args.work_counter = work_counter;
        args.sg_ptr = sg_device_ptr;
        args.tiles_X = tiles_X;
        args.tiles_Y = tiles_Y;
        args.total_tiles = total_tiles;

        const dim3 grid(num_persistent);
        tk_v5_col_only::persistent_quantize_phase2_col_only_kernel<true>
            <<<grid, V3_THREADS, s_col_only_dshmem, stream>>>(
                tmap_in, tmap_out_t, tmap_sc_col,
                (size_t)M, (size_t)K, args);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_col_only failed: ", cudaGetErrorString(err));

    auto col_fp4_v = col_fp4.view(torch::kFloat4_e2m1fn_x2);
    auto col_sc_v  = col_sc.view(torch::kFloat8_e4m3fn);
    return std::make_tuple(col_fp4_v, col_sc_v, sg_tensor, sg_tensor);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_mxfp4_row_nvfp4_col(
    torch::Tensor input,
    int64_t threads,
    bool data_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);
    TORCH_CHECK(threads == 128 || threads == 256,
                "mixed MXFP4/NVFP4 producer threads must be 128 or 256");

    auto device = input.device();
    auto row_fp4 = torch::empty(
        {M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto sg = torch::zeros(
        {1}, torch::dtype(torch::kFloat32).device(device));

    const int blocks_x = static_cast<int>(K / 128);
    const int total_tiles = static_cast<int>((M / 128) * blocks_x);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto rng_state = data_stochastic_rounding
        ? tk_make_advancing_rng_state(
            input, rng_seed, rng_subsequence, stream)
        : torch::Tensor();
    const auto* rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
#define LAUNCH_MXFP4_MIXED_ROW(DATA_SR) \
    mxfp4_row_native_sg_kernel<DATA_SR> \
        <<<total_tiles, static_cast<int>(threads), 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
            reinterpret_cast<uint8_t*>(row_fp4.data_ptr()), \
            row_sc.data_ptr<uint8_t>(), \
            sg.data_ptr<float>(), \
            static_cast<int>(M), \
            static_cast<int>(K), \
            blocks_x, \
            total_tiles, \
            rng_state_ptr)
    if (data_stochastic_rounding) {
        LAUNCH_MXFP4_MIXED_ROW(true);
    } else {
        LAUNCH_MXFP4_MIXED_ROW(false);
    }
#undef LAUNCH_MXFP4_MIXED_ROW
    auto err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "mixed MXFP4 row/native-SG producer failed: ",
        cudaGetErrorString(err));

    auto col = tk_quantize_col_only(input, sg);
    return std::make_tuple(
        row_fp4,
        row_sc,
        std::get<0>(col),
        std::get<1>(col),
        sg);
}


// ═══════════════════════════════════════════════════════════════════
// 1b. CUDA Graph-safe alloc/launch split for tk_v4_quantize_for_gemm
// ═══════════════════════════════════════════════════════════════════

// _alloc: pre-create all output tensors + sync buffers. Call OUTSIDE graph capture.
// Returns: (row_fp4, row_sc, col_fp4, col_sc, amax_buf, sync_buf)
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_alloc(int64_t M, int64_t K, bool return_transpose, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);
    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto col_sc  = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_u8) : torch::empty({0}, opts_u8);
    auto amax_buf = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_u32);

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, amax_buf, sync_buf);
}

// _launch: kernel-only dispatch — NO allocations, safe inside CUDA graph capture.
// Returns: (row_fp4_view, row_sc_view, col_fp4_view, col_sc_view, sg, amax)
std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_v4_quantize_for_gemm_launch(
    torch::Tensor input, bool return_transpose, bool encode_centric,
    // Pre-allocated buffers from _alloc:
    torch::Tensor row_fp4, torch::Tensor row_sc,
    torch::Tensor col_fp4, torch::Tensor col_sc,
    torch::Tensor amax_buf, torch::Tensor sync_buf
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;
    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());

    // Graph-safe: cudaMemsetAsync
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

    nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    nvfp4_scale_t *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    using namespace tk_v3;
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached_info();
    const int v3_dshmem = return_transpose ? ci.v3_dshmem_t : ci.v3_dshmem;

    if (total_tiles < PERSISTENT_THRESHOLD || disable_persistent_quant_for_conn1()) {
        const int max_bps = return_transpose ? ci.v3_max_bps_t : ci.v3_max_bps;
        const int max_concurrent = max_bps * persistent_launch_sms(ci);
        const bool can_fuse = (total_tiles <= max_concurrent && max_bps > 0);

        if (can_fuse) {
            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
            create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
            unsigned int *done = sync_data + 2, *ready = sync_data + 3;
            if (encode_centric) {
                if (return_transpose)
                    launch_v3<true, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, true>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v3<true, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v3<false, false>(tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_ptr, sg_ptr, done, ready, M, K, scale_stride, scale_stride_t, stream);
            }
        } else {
            int64_t n = M * K;
            int amax_blocks = pipelined_amax::grid_size(n);
            int amax_smem = pipelined_amax::smem_size();
            pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
                amax_ptr, sg_ptr, sync_data, n);
            alignas(64) CUtensorMap ti{}, to{}, tot{};
            create_tma_2d(ti, input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
            create_tma_2d(to, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
            if (return_transpose)
                create_tma_2d(tot, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);
            if (encode_centric) {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, true>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            } else {
                if (return_transpose)
                    launch_v2_kernel<false, false, true, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
                else
                    launch_v2_kernel<false, false, false, false>(ti, to, tot, sc_ptr, sc_t_ptr, amax_ptr, amax_ptr, M, K, scale_stride, scale_stride_t, stream);
            }
        }
    } else {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        if (return_transpose)
            create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        if (return_transpose && sc_t_ptr) {
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }

        const int p_max_bps = return_transpose ? ci.v4_max_bps_t : ci.v4_max_bps;
        int num_persistent = p_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;
        pargs.col_scales_ptr = sc_t_ptr;
        pargs.col_scale_stride = scale_stride_t;
        pargs.swizzle_scales = true;

        const dim3 grid(num_persistent);
        if (encode_centric) {
            if (return_transpose)
                tk_v5::persistent_quantize_kernel<true, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            else
                tk_v5::persistent_quantize_kernel<false, true><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
        } else {
            if (return_transpose)
                tk_v5::persistent_quantize_kernel<true, false><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
            else
                tk_v5::persistent_quantize_kernel<false, false><<<grid, V3_THREADS, v3_dshmem, stream>>>(
                    tmap_in, tmap_out, tmap_out_t, tmap_sc_row, tmap_sc_col,
                    sc_ptr, M, K, scale_stride, pargs);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_v4_quantize_for_gemm_launch failed: ", cudaGetErrorString(err));

    auto row_fp4_v = row_fp4.view(torch::kFloat4_e2m1fn_x2);
    auto row_sc_v  = row_sc.view(torch::kFloat8_e4m3fn);
    auto col_fp4_v = return_transpose ? col_fp4.view(torch::kFloat4_e2m1fn_x2) : col_fp4;
    auto col_sc_v  = return_transpose ? col_sc.view(torch::kFloat8_e4m3fn) : col_sc;
    return std::make_tuple(row_fp4_v, row_sc_v, col_fp4_v, col_sc_v,
                           amax_buf.narrow(0, 1, 1), amax_buf.narrow(0, 1, 1));
}


// ═══════════════════════════════════════════════════════════════════
// 2. tk_quantize_transpose — pre-computed amax (v2 kernel)
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_transpose(
    torch::Tensor input, torch::Tensor amax_row, torch::Tensor amax_col, bool return_transpose,
    bool stochastic_rounding,
    bool encode_centric,
    bool scale_stochastic_rounding,
    bool row_rht,
    bool col_rht,
    bool return_row,
    torch::Tensor rng_state
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(amax_row.is_cuda() && amax_row.scalar_type() == torch::kFloat32 && amax_row.numel() == 1);
    TORCH_CHECK(return_row || return_transpose, "tk_quantize_transpose must produce row or transpose output");

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 32 == 0 && K % 32 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc  = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto col_sc  = return_transpose ? torch::empty({K, scale_stride_t}, opts_u8) : torch::empty({0}, opts_u8);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    if (return_transpose)
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    auto *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    auto *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;

    const float *amax_r = reinterpret_cast<const float*>(amax_row.data_ptr());
    const float *amax_c = (amax_col.numel() > 0) ? reinterpret_cast<const float*>(amax_col.data_ptr()) : amax_r;
    const size_t *rng_state_ptr = nullptr;
    if (rng_state.defined()) {
        TORCH_CHECK(
            rng_state.is_cuda() && rng_state.is_contiguous() &&
                rng_state.scalar_type() == torch::kInt64 && rng_state.numel() == 2,
            "rng_state must be a contiguous CUDA int64 tensor with shape [2]");
        TORCH_CHECK(
            rng_state.get_device() == input.get_device(),
            "rng_state must be on the same CUDA device as input");
        rng_state_ptr = reinterpret_cast<const size_t *>(rng_state.data_ptr<int64_t>());
    }

#define TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, ROW_RHT_FLAG, COL_RHT_FLAG, RETURN_ROW_FLAG) \
    launch_v2_kernel<SR_FLAG, false, RT_FLAG, ENCODE_FLAG, SCALE_SR_FLAG, ROW_RHT_FLAG,    \
                     COL_RHT_FLAG, RETURN_ROW_FLAG>(                                      \
        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,              \
        scale_stride, scale_stride_t, stream, nullptr, rng_state_ptr)

#define TK_LAUNCH_V2_RHT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG)                     \
    do {                                                                                   \
        if (row_rht) {                                                                     \
            if (col_rht) {                                                                 \
                if (return_row) TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, true, true); \
                else TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, true, false); \
            } else {                                                                       \
                if (return_row) TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, false, true); \
                else TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, false, false); \
            }                                                                              \
        } else {                                                                           \
            if (col_rht) {                                                                 \
                if (return_row) TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, true, true); \
                else TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, true, false); \
            } else {                                                                       \
                if (return_row) TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, false, true); \
                else TK_LAUNCH_V2(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, false, false); \
            }                                                                              \
        }                                                                                  \
    } while (0)

    if (stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) {
                if (return_transpose) TK_LAUNCH_V2_RHT(true, true, true, true);
                else TK_LAUNCH_V2_RHT(true, true, false, true);
            } else {
                if (return_transpose) TK_LAUNCH_V2_RHT(true, true, true, false);
                else TK_LAUNCH_V2_RHT(true, true, false, false);
            }
        } else if (encode_centric) {
            if (return_transpose) TK_LAUNCH_V2_RHT(true, false, true, true);
            else TK_LAUNCH_V2_RHT(true, false, false, true);
        } else {
            if (return_transpose) TK_LAUNCH_V2_RHT(true, false, true, false);
            else TK_LAUNCH_V2_RHT(true, false, false, false);
        }
    } else if (scale_stochastic_rounding) {
        if (encode_centric) {
            if (return_transpose) TK_LAUNCH_V2_RHT(false, true, true, true);
            else TK_LAUNCH_V2_RHT(false, true, false, true);
        } else {
            if (return_transpose) TK_LAUNCH_V2_RHT(false, true, true, false);
            else TK_LAUNCH_V2_RHT(false, true, false, false);
        }
    } else if (encode_centric) {
        if (return_transpose) TK_LAUNCH_V2_RHT(false, false, true, true);
        else TK_LAUNCH_V2_RHT(false, false, false, true);
    } else {
        if (return_transpose) TK_LAUNCH_V2_RHT(false, false, true, false);
        else TK_LAUNCH_V2_RHT(false, false, false, false);
    }

#undef TK_LAUNCH_V2_RHT
#undef TK_LAUNCH_V2

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_transpose failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

static void tk_quantize_transpose_out(
    torch::Tensor input,
    torch::Tensor amax_row,
    torch::Tensor amax_col,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    bool return_transpose,
    bool stochastic_rounding,
    bool encode_centric,
    bool scale_stochastic_rounding,
    bool row_rht,
    bool col_rht,
    bool return_row
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(amax_row.is_cuda() && amax_row.scalar_type() == torch::kFloat32 && amax_row.numel() == 1);
    TORCH_CHECK(amax_col.is_cuda() && amax_col.scalar_type() == torch::kFloat32 && amax_col.numel() == 1);
    TORCH_CHECK(return_row || return_transpose, "tk_quantize_transpose_out must produce row or transpose output");

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 32 == 0 && K % 32 == 0);
    if (return_row) {
        TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous() && row_fp4.numel() >= M * (K / 2));
        TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    }
    if (return_transpose) {
        TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous() && col_fp4.numel() >= K * (M / 2));
        TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    if (return_row)
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    if (return_transpose)
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    auto *sc_ptr   = return_row ? reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()) : nullptr;
    auto *sc_t_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;
    const float *amax_r = reinterpret_cast<const float*>(amax_row.data_ptr());
    const float *amax_c = reinterpret_cast<const float*>(amax_col.data_ptr());

#define TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, ROW_RHT_FLAG, COL_RHT_FLAG, RETURN_ROW_FLAG) \
    launch_v2_kernel<SR_FLAG, false, RT_FLAG, ENCODE_FLAG, SCALE_SR_FLAG, ROW_RHT_FLAG, \
                     COL_RHT_FLAG, RETURN_ROW_FLAG>(                                   \
        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,          \
        scale_stride, scale_stride_t, stream)

#define TK_LAUNCH_V2_OUT_RHT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG)                 \
    do {                                                                                   \
        if (row_rht) {                                                                     \
            if (col_rht) {                                                                 \
                if (return_row) TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, true, true); \
                else TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, true, false); \
            } else {                                                                       \
                if (return_row) TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, false, true); \
                else TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, true, false, false); \
            }                                                                              \
        } else {                                                                           \
            if (col_rht) {                                                                 \
                if (return_row) TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, true, true); \
                else TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, true, false); \
            } else {                                                                       \
                if (return_row) TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, false, true); \
                else TK_LAUNCH_V2_OUT(SR_FLAG, SCALE_SR_FLAG, RT_FLAG, ENCODE_FLAG, false, false, false); \
            }                                                                              \
        }                                                                                  \
    } while (0)

    if (stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) {
                if (return_transpose) TK_LAUNCH_V2_OUT_RHT(true, true, true, true);
                else TK_LAUNCH_V2_OUT_RHT(true, true, false, true);
            } else {
                if (return_transpose) TK_LAUNCH_V2_OUT_RHT(true, true, true, false);
                else TK_LAUNCH_V2_OUT_RHT(true, true, false, false);
            }
        } else if (encode_centric) {
            if (return_transpose) TK_LAUNCH_V2_OUT_RHT(true, false, true, true);
            else TK_LAUNCH_V2_OUT_RHT(true, false, false, true);
        } else {
            if (return_transpose) TK_LAUNCH_V2_OUT_RHT(true, false, true, false);
            else TK_LAUNCH_V2_OUT_RHT(true, false, false, false);
        }
    } else if (scale_stochastic_rounding) {
        if (encode_centric) {
            if (return_transpose) TK_LAUNCH_V2_OUT_RHT(false, true, true, true);
            else TK_LAUNCH_V2_OUT_RHT(false, true, false, true);
        } else {
            if (return_transpose) TK_LAUNCH_V2_OUT_RHT(false, true, true, false);
            else TK_LAUNCH_V2_OUT_RHT(false, true, false, false);
        }
    } else if (encode_centric) {
        if (return_transpose) TK_LAUNCH_V2_OUT_RHT(false, false, true, true);
        else TK_LAUNCH_V2_OUT_RHT(false, false, false, true);
    } else {
        if (return_transpose) TK_LAUNCH_V2_OUT_RHT(false, false, true, false);
        else TK_LAUNCH_V2_OUT_RHT(false, false, false, false);
    }

#undef TK_LAUNCH_V2_OUT_RHT
#undef TK_LAUNCH_V2_OUT

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_transpose_out failed: ", cudaGetErrorString(err));
}

static torch::Tensor tk_compute_amax_sg(torch::Tensor input, cudaStream_t stream) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto amax_buf = torch::empty({3}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    auto *block_count = reinterpret_cast<unsigned int*>(amax_ptr + 2);
    cudaMemsetAsync(amax_ptr, 0, 3 * sizeof(float), stream);

    const int64_t n = input.numel();
    const int amax_blocks = pipelined_amax::grid_size(n);
    const int amax_smem = pipelined_amax::smem_size();
    pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        amax_ptr, sg_ptr, block_count, n);
    return amax_buf;
}

__global__ void tk_rescale_amax_sg_kernel(float *amax_sg, float factor) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        amax_sg[0] *= factor;
        amax_sg[1] *= factor;
    }
}

static void tk_apply_global_scale_target(
    torch::Tensor amax_sg,
    float global_scale_target,
    cudaStream_t stream
) {
    if (global_scale_target == kDefaultNvfp4GlobalScaleTarget) {
        return;
    }
    // Existing quantizers use 448 internally. Rescaling both values makes the
    // hard-coded path equivalent to amax / (6 * global_scale_target).
    const float factor = kDefaultNvfp4GlobalScaleTarget / global_scale_target;
    tk_rescale_amax_sg_kernel<<<1, 1, 0, stream>>>(amax_sg.data_ptr<float>(), factor);
}

static std::tuple<torch::Tensor, torch::Tensor> tk_apply_rht16_bf16_amax(
    torch::Tensor input,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    cudaStream_t stream
) {
    auto out = torch::empty_like(input);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto amax_buf = torch::empty({2}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);

    const int64_t M = input.size(0), K = input.size(1);
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((groups + threads - 1) / threads, 65535));
    const int smem = pipelined_amax::WARPS * sizeof(float);
    rht16_bf16_amax_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        amax_ptr,
        sg_ptr,
        M, K, across_cols, with_random_sign_mask, rng_seed, rng_subsequence);
    return std::make_tuple(out, amax_buf);
}

static torch::Tensor tk_scan_rht16_amax(
    torch::Tensor input,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    cudaStream_t stream
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto amax_buf = torch::empty({2}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);

    const int64_t M = input.size(0), K = input.size(1);
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((groups + threads - 1) / threads, 65535));
    const int smem = pipelined_amax::WARPS * sizeof(float);
    rht16_bf16_amax_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        nullptr,
        amax_ptr,
        sg_ptr,
        M, K, across_cols, with_random_sign_mask, rng_seed, rng_subsequence);
    return amax_buf;
}

static std::tuple<torch::Tensor, torch::Tensor> tk_scan_rht16_and_orig_amax(
    torch::Tensor input,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    cudaStream_t stream
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto rht_amax_buf = torch::empty({2}, opts_f32);
    auto orig_amax_buf = torch::empty({2}, opts_f32);
    float *rht_amax_ptr = rht_amax_buf.data_ptr<float>();
    float *orig_amax_ptr = orig_amax_buf.data_ptr<float>();
    cudaMemsetAsync(rht_amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(orig_amax_ptr, 0, 2 * sizeof(float), stream);

    const int64_t M = input.size(0), K = input.size(1);
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((groups + threads - 1) / threads, 65535));
    const int smem = 2 * pipelined_amax::WARPS * sizeof(float);
    rht16_bf16_amax_with_orig_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        nullptr,
        rht_amax_ptr,
        rht_amax_ptr + 1,
        orig_amax_ptr,
        orig_amax_ptr + 1,
        M, K, across_cols, with_random_sign_mask, rng_seed, rng_subsequence);
	    return std::make_tuple(rht_amax_buf, orig_amax_buf);
}

static std::tuple<torch::Tensor, torch::Tensor> tk_scan_rmsnorm_rht16_and_orig_amax(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    bool across_cols,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    cudaStream_t stream
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto rht_amax_buf = torch::empty({2}, opts_f32);
    auto orig_amax_buf = torch::empty({2}, opts_f32);
    float *rht_amax_ptr = rht_amax_buf.data_ptr<float>();
    float *orig_amax_ptr = orig_amax_buf.data_ptr<float>();
    cudaMemsetAsync(rht_amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(orig_amax_ptr, 0, 2 * sizeof(float), stream);

    const int64_t M = input.size(0), K = input.size(1);
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((groups + threads - 1) / threads, 65535));
    const int smem = 2 * pipelined_amax::WARPS * sizeof(float);
    rmsnorm_rht16_amax_with_orig_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        inv_rms.data_ptr<float>(),
        rht_amax_ptr,
        rht_amax_ptr + 1,
        orig_amax_ptr,
        orig_amax_ptr + 1,
        M, K, across_cols, with_random_sign_mask, rng_seed, rng_subsequence);
    return std::make_tuple(rht_amax_buf, orig_amax_buf);
}

static torch::Tensor tk_prepare_rmsnorm_bf16_amax(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    float epsilon,
    bool compute_inv_rms,
    cudaStream_t stream
) {
    auto result = torch::empty(
        {2}, torch::dtype(torch::kFloat32).device(input.device()));
    cudaMemsetAsync(result.data_ptr<float>(), 0, 2 * sizeof(float), stream);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(input.size(0), 65535));
    if (compute_inv_rms) {
        rmsnorm_bf16_amax_kernel<true><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
            inv_rms.data_ptr<float>(),
            result.data_ptr<float>(),
            epsilon,
            input.size(0),
            input.size(1));
    } else {
        rmsnorm_bf16_amax_kernel<false><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
            inv_rms.data_ptr<float>(),
            result.data_ptr<float>(),
            epsilon,
            input.size(0),
            input.size(1));
    }
    compute_sg_kernel<<<1, 32, 0, stream>>>(
        result.data_ptr<float>(), result.data_ptr<float>() + 1, 1);
    return result;
}

static torch::Tensor tk_compute_sqrelu_amax_sg(
    torch::Tensor input,
    torch::Tensor aux,
    bool deriv,
    cudaStream_t stream
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto amax_buf = torch::empty({2}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);

    const int64_t n = input.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 65535));
    const int smem = pipelined_amax::WARPS * sizeof(float);
    sqrelu_amax_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        deriv ? reinterpret_cast<const __nv_bfloat16*>(aux.data_ptr()) : nullptr,
        amax_ptr,
        sg_ptr,
        n,
        deriv);
    return amax_buf;
}

static std::tuple<torch::Tensor, torch::Tensor> tk_scan_sqrelu_rht16_and_orig_amax(
    torch::Tensor input,
    torch::Tensor aux,
    bool across_cols,
    bool deriv,
    cudaStream_t stream
) {
    auto opts_f32 = torch::dtype(torch::kFloat32).device(input.device());
    auto rht_amax_buf = torch::empty({2}, opts_f32);
    auto orig_amax_buf = torch::empty({2}, opts_f32);
    float *rht_amax_ptr = rht_amax_buf.data_ptr<float>();
    float *orig_amax_ptr = orig_amax_buf.data_ptr<float>();
    cudaMemsetAsync(rht_amax_ptr, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(orig_amax_ptr, 0, 2 * sizeof(float), stream);

    const int64_t M = input.size(0), K = input.size(1);
    const int64_t groups = across_cols ? (M * (K / 16)) : ((M / 16) * K);
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>((groups + threads - 1) / threads, 65535));
    const int smem = 2 * pipelined_amax::WARPS * sizeof(float);
    sqrelu_rht16_amax_with_orig_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        deriv ? reinterpret_cast<const __nv_bfloat16*>(aux.data_ptr()) : nullptr,
        rht_amax_ptr,
        rht_amax_ptr + 1,
        orig_amax_ptr,
        orig_amax_ptr + 1,
        M, K, across_cols, deriv);
    return std::make_tuple(rht_amax_buf, orig_amax_buf);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_for_gemm_opt(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    float global_scale_target,
    std::string data_sr_axes
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before v5 opt quantization: ",
                cudaGetErrorString(set_device_err));
    auto stream = at::cuda::getCurrentCUDAStream();
    V5QuantSequenceGuard sequence_guard(stream);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);
    TORCH_CHECK(M % 16 == 0 && K % 16 == 0);
    TORCH_CHECK(
        std::isfinite(global_scale_target) && global_scale_target > 0.0f &&
            global_scale_target <= kMaxNvfp4GlobalScaleTarget,
        "global_scale_target must be finite and in (0, 512], got ", global_scale_target);

    for (char &c : rht_axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    TORCH_CHECK(
        rht_axes == "none" || rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
        rht_axes == "rowcol" || rht_axes == "row_col",
        "Unsupported regular TK RHT axes: ", rht_axes);
    TORCH_CHECK(
        global_scale_target == kDefaultNvfp4GlobalScaleTarget || (!row_rht && !col_rht),
        "non-default global_scale_target is currently supported only with rht_axes='none'");

    for (char &c : data_sr_axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    if (data_sr_axes == "all" || data_sr_axes == "rowcol" || data_sr_axes == "row_col") {
        data_sr_axes = "both";
    } else if (data_sr_axes == "column" || data_sr_axes == "columns" || data_sr_axes == "wgrad") {
        data_sr_axes = "col";
    } else if (data_sr_axes == "dgrad") {
        data_sr_axes = "row";
    } else if (data_sr_axes == "off" || data_sr_axes == "0") {
        data_sr_axes = "none";
    }
    TORCH_CHECK(
        data_sr_axes == "none" || data_sr_axes == "row" ||
        data_sr_axes == "col" || data_sr_axes == "both",
        "Unsupported regular TK data SR axes: ", data_sr_axes);
    const bool row_data_sr = data_stochastic_rounding &&
        (data_sr_axes == "row" || data_sr_axes == "both");
    const bool col_data_sr = data_stochastic_rounding &&
        (data_sr_axes == "col" || data_sr_axes == "both");

    if (!row_rht && !col_rht && !row_data_sr && !col_data_sr && !scale_stochastic_rounding &&
        global_scale_target == kDefaultNvfp4GlobalScaleTarget) {
        auto base = tk_v4_quantize_for_gemm(input, return_transpose, encode_centric);
        return std::make_tuple(
            std::get<0>(base), std::get<1>(base), std::get<2>(base),
            std::get<3>(base), std::get<4>(base), std::get<5>(base));
    }

    const bool use_stochastic_rounding =
        row_data_sr || col_data_sr || scale_stochastic_rounding;
    auto make_rng_state = [&]() {
        return use_stochastic_rounding
            ? tk_make_advancing_rng_state(input, rng_seed, rng_subsequence, stream)
            : torch::Tensor();
    };
    const bool fuse_rht_in_quant = !with_random_sign_mask && !disable_fused_rht_quant();
    torch::Tensor row_input = input;
    torch::Tensor row_amax;
    torch::Tensor orig_amax_from_row_rht_scan;
    bool row_quant_rht = false;
    if (row_rht) {
        if (fuse_rht_in_quant) {
            if (return_transpose && !col_rht) {
                auto scans = tk_scan_rht16_and_orig_amax(
                    input, true, false, rng_seed, rng_subsequence, stream);
                row_amax = std::get<0>(scans);
                orig_amax_from_row_rht_scan = std::get<1>(scans);
            } else {
                row_amax = tk_scan_rht16_amax(
                    input, true, false, rng_seed, rng_subsequence, stream);
            }
            row_quant_rht = true;
        } else {
            auto row_rht_out = tk_apply_rht16_bf16_amax(
                input, true, with_random_sign_mask, rng_seed, rng_subsequence, stream);
            row_input = std::get<0>(row_rht_out);
            row_amax = std::get<1>(row_rht_out);
        }
    } else {
        row_amax = tk_compute_amax_sg(row_input, stream);
        tk_apply_global_scale_target(row_amax, global_scale_target, stream);
    }

    if (return_transpose && row_input.data_ptr() == input.data_ptr() &&
        row_data_sr == col_data_sr) {
        torch::Tensor col_input = input;
        torch::Tensor col_amax;
        bool col_quant_rht = false;
        if (col_rht) {
            if (fuse_rht_in_quant) {
                col_amax = tk_scan_rht16_amax(
                    input, false, false, rng_seed, rng_subsequence, stream);
                col_quant_rht = true;
            } else {
                auto col_rht_out = tk_apply_rht16_bf16_amax(
                    input, false, with_random_sign_mask, rng_seed, rng_subsequence, stream);
                col_input = std::get<0>(col_rht_out);
                col_amax = std::get<1>(col_rht_out);
            }
        } else if (row_rht) {
            col_amax = orig_amax_from_row_rht_scan.defined()
                ? orig_amax_from_row_rht_scan
                : tk_compute_amax_sg(col_input, stream);
        } else {
            col_amax = row_amax;
        }

        if (col_input.data_ptr() == input.data_ptr()) {
            auto rng_state = make_rng_state();
            auto quant = tk_quantize_transpose(
                input,
                row_amax.narrow(0, 0, 1),
                col_amax.narrow(0, 0, 1),
                true,
                row_data_sr,
                encode_centric,
                scale_stochastic_rounding,
                row_quant_rht,
                col_quant_rht,
                true,
                rng_state);

            auto err = cudaGetLastError();
            TORCH_CHECK(err == cudaSuccess, "tk_quantize_for_gemm_opt failed: ", cudaGetErrorString(err));

            return std::make_tuple(
                std::get<0>(quant).view(torch::kFloat4_e2m1fn_x2),
                std::get<1>(quant)
                    .view(torch::kFloat8_e4m3fn)
                    .view({M / 128, K / 64, 512}),
                std::get<2>(quant).view(torch::kFloat4_e2m1fn_x2),
                std::get<3>(quant)
                    .view(torch::kFloat8_e4m3fn)
                    .view({K / 128, M / 64, 512}),
                row_amax.narrow(0, 1, 1),
                col_amax.narrow(0, 1, 1)
            );
        }
    }

    auto row_rng_state = make_rng_state();
    auto row_quant = tk_quantize_transpose(
        row_input,
        row_amax.narrow(0, 0, 1),
        row_amax.narrow(0, 0, 1),
        false,
        row_data_sr,
        encode_centric,
        scale_stochastic_rounding,
        row_quant_rht,
        false,
        true,
        row_rng_state);

    torch::Tensor col_fp4 = torch::empty({0}, torch::dtype(torch::kUInt8).device(input.device()));
    torch::Tensor col_sc = torch::empty({0}, torch::dtype(torch::kUInt8).device(input.device()));
    torch::Tensor col_sg = row_amax.narrow(0, 1, 1);
    if (return_transpose) {
        torch::Tensor col_input = input;
        torch::Tensor col_amax;
        bool col_quant_rht = false;
        if (col_rht) {
            if (fuse_rht_in_quant) {
                col_amax = tk_scan_rht16_amax(
                    input, false, false, rng_seed, rng_subsequence, stream);
                col_quant_rht = true;
            } else {
                auto col_rht_out = tk_apply_rht16_bf16_amax(
                    input, false, with_random_sign_mask, rng_seed, rng_subsequence, stream);
                col_input = std::get<0>(col_rht_out);
                col_amax = std::get<1>(col_rht_out);
            }
        } else if (row_rht) {
            col_amax = orig_amax_from_row_rht_scan.defined()
                ? orig_amax_from_row_rht_scan
                : tk_compute_amax_sg(col_input, stream);
        } else {
            col_amax = row_amax;
        }
        auto col_rng_state = make_rng_state();
        auto col_quant = tk_quantize_transpose(
            col_input,
            col_amax.narrow(0, 0, 1),
            col_amax.narrow(0, 0, 1),
            true,
            col_data_sr,
            encode_centric,
            scale_stochastic_rounding,
            false,
            col_quant_rht,
            false,
            col_rng_state);
        col_fp4 = std::get<2>(col_quant);
        col_sc = std::get<3>(col_quant);
        col_sg = col_amax.narrow(0, 1, 1);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_for_gemm_opt failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        std::get<0>(row_quant).view(torch::kFloat4_e2m1fn_x2),
        std::get<1>(row_quant)
            .view(torch::kFloat8_e4m3fn)
            .view({M / 128, K / 64, 512}),
        return_transpose ? col_fp4.view(torch::kFloat4_e2m1fn_x2) : col_fp4,
        return_transpose
            ? col_sc.view(torch::kFloat8_e4m3fn)
                  .view({K / 128, M / 64, 512})
            : col_sc,
        row_amax.narrow(0, 1, 1),
        col_sg
    );
}

template <bool ENCODE_CENTRIC>
static void launch_sqrelu_row_rht_persistent(
    torch::Tensor input,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    const uint64_t *rng_state_ptr,
    cudaStream_t stream) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    using namespace tk_v3;

    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    static int s_dshmem = -1;
    static int s_max_bps = -1;
    if (s_dshmem < 0) {
        s_dshmem = persistent_sqrelu_deriv_quant::persistent_sqrelu_quant_smem_size<true>();
        auto kernel =
            persistent_sqrelu_deriv_quant::persistent_sqrelu_quantize_kernel<
                true, true, false, ENCODE_CENTRIC, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, s_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&s_max_bps, kernel, V3_THREADS, s_dshmem);
    }

    const auto& ci = get_cached_info();
    int num_persistent = s_max_bps * persistent_launch_sms(ci);
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    TORCH_CHECK(num_persistent > 0, "sqrelu row-RHT persistent launch has zero resident CTAs");

    cudaMemsetAsync(amaxes.data_ptr<float>(), 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_buf.data_ptr<int>(), 0, 4 * sizeof(unsigned int), stream);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto* sync_ptr = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int>());
    float* amax_ptr = amaxes.data_ptr<float>();

    tk_v5::PersistentArgs pargs;
    memset(&pargs, 0, sizeof(pargs));
    pargs.work_counter_phase1 = sync_ptr;
    pargs.work_counter_phase2 = sync_ptr + 1;
    pargs.global_amax = amax_ptr;
    pargs.done_counter = sync_ptr + 2;
    pargs.ready_flag = sync_ptr + 3;
    pargs.tiles_X = tiles_X;
    pargs.tiles_Y = tiles_Y;
    pargs.total_tiles = total_tiles;
    pargs.num_persistent = num_persistent;
    pargs.sg_output = sgs.data_ptr<float>();

    const dim3 grid(num_persistent);
    persistent_sqrelu_deriv_quant::persistent_sqrelu_quantize_kernel<
        true, true, false, ENCODE_CENTRIC, false>
        <<<grid, V3_THREADS, s_dshmem, stream>>>(
            tmap_in,
            tmap_out,
            tmap_out_t,
            tmap_sc_row,
            tmap_sc_col,
            M,
            K,
            scale_stride,
            pargs,
            amax_ptr + 1,
            0,
            0,
            rng_state_ptr);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor>
tk_sqrelu_quantize_row_rht_persistent(
    torch::Tensor input,
    bool encode_centric) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_u8);
    auto col_sc = torch::empty({K, scale_stride_t}, opts_u8);
    auto sgs = torch::empty({2}, opts_f32);
    auto amaxes = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (encode_centric) {
        launch_sqrelu_row_rht_persistent<true>(
            input, row_fp4, row_sc, col_fp4, col_sc, sgs, amaxes, sync_buf, nullptr, stream);
    } else {
        launch_sqrelu_row_rht_persistent<false>(
            input, row_fp4, row_sc, col_fp4, col_sc, sgs, amaxes, sync_buf, nullptr, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_sqrelu_quantize_row_rht_persistent failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4.view(torch::kFloat4_e2m1fn_x2),
        row_sc.view(torch::kFloat8_e4m3fn),
        col_fp4.view(torch::kFloat4_e2m1fn_x2),
        col_sc.view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 0, 1),
        sgs.narrow(0, 1, 1));
}

template <bool DATA_SR, bool ENCODE_CENTRIC, bool SCALE_SR>
static void launch_sqrelu_deriv_row_rht_persistent(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    const uint64_t *rng_state_ptr,
    cudaStream_t stream) {
    const int64_t M = dh.size(0);
    const int64_t K = dh.size(1);
    using namespace tk_v3;

    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    static int s_dshmem = -1;
    static int s_max_bps = -1;
    if (s_dshmem < 0) {
        s_dshmem = persistent_sqrelu_deriv_quant::persistent_sqrelu_deriv_quant_smem_size<true>();
        auto kernel =
            persistent_sqrelu_deriv_quant::persistent_sqrelu_deriv_quantize_kernel<
                true, true, DATA_SR, ENCODE_CENTRIC, SCALE_SR>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, s_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&s_max_bps, kernel, V3_THREADS, s_dshmem);
    }

    const auto& ci = get_cached_info();
    int num_persistent = s_max_bps * persistent_launch_sms(ci);
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    TORCH_CHECK(num_persistent > 0, "sqrelu derivative row-RHT persistent launch has zero resident CTAs");

    cudaMemsetAsync(amaxes.data_ptr<float>(), 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_buf.data_ptr<int>(), 0, 4 * sizeof(unsigned int), stream);

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto* sync_ptr = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int>());
    float* amax_ptr = amaxes.data_ptr<float>();

    tk_v5::PersistentArgs pargs;
    memset(&pargs, 0, sizeof(pargs));
    pargs.work_counter_phase1 = sync_ptr;
    pargs.work_counter_phase2 = sync_ptr + 1;
    pargs.global_amax = amax_ptr;
    pargs.done_counter = sync_ptr + 2;
    pargs.ready_flag = sync_ptr + 3;
    pargs.tiles_X = tiles_X;
    pargs.tiles_Y = tiles_Y;
    pargs.total_tiles = total_tiles;
    pargs.num_persistent = num_persistent;
    pargs.sg_output = sgs.data_ptr<float>();

    const dim3 grid(num_persistent);
    persistent_sqrelu_deriv_quant::persistent_sqrelu_deriv_quantize_kernel<
        true, true, DATA_SR, ENCODE_CENTRIC, SCALE_SR>
        <<<grid, V3_THREADS, s_dshmem, stream>>>(
            tmap_dh,
            tmap_h1,
            tmap_out,
            tmap_out_t,
            tmap_sc_row,
            tmap_sc_col,
            M,
            K,
            scale_stride,
            pargs,
            amax_ptr + 1,
            0,
            0,
            rng_state_ptr);
}

static void launch_sqrelu_deriv_row_rht_persistent_dispatch(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    bool data_stochastic_rounding,
    bool encode_centric,
    bool scale_stochastic_rounding,
    const uint64_t *rng_state_ptr,
    cudaStream_t stream) {
#define TK_LAUNCH_SQRELU_DERIV_ROW_RHT(SR_FLAG, ENCODE_FLAG, SCALE_SR_FLAG) \
    launch_sqrelu_deriv_row_rht_persistent<SR_FLAG, ENCODE_FLAG, SCALE_SR_FLAG>( \
        dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, sgs, amaxes, sync_buf, \
        rng_state_ptr, stream)

    if (data_stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) TK_LAUNCH_SQRELU_DERIV_ROW_RHT(true, true, true);
            else TK_LAUNCH_SQRELU_DERIV_ROW_RHT(true, false, true);
        } else if (encode_centric) {
            TK_LAUNCH_SQRELU_DERIV_ROW_RHT(true, true, false);
        } else {
            TK_LAUNCH_SQRELU_DERIV_ROW_RHT(true, false, false);
        }
    } else if (scale_stochastic_rounding) {
        if (encode_centric) TK_LAUNCH_SQRELU_DERIV_ROW_RHT(false, true, true);
        else TK_LAUNCH_SQRELU_DERIV_ROW_RHT(false, false, true);
    } else if (encode_centric) {
        TK_LAUNCH_SQRELU_DERIV_ROW_RHT(false, true, false);
    } else {
        TK_LAUNCH_SQRELU_DERIV_ROW_RHT(false, false, false);
    }

#undef TK_LAUNCH_SQRELU_DERIV_ROW_RHT
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor>
tk_sqrelu_deriv_quantize_row_rht_persistent(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence) {
    const int64_t M = dh.size(0);
    const int64_t K = dh.size(1);
    auto device = dh.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_u8);
    auto col_sc = torch::empty({K, scale_stride_t}, opts_u8);
    auto sgs = torch::empty({2}, opts_f32);
    auto amaxes = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto rng_state = (data_stochastic_rounding || scale_stochastic_rounding)
        ? tk_make_advancing_rng_state(dh, rng_seed, rng_subsequence, stream)
        : torch::Tensor();
    const auto *rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t *>(rng_state.data_ptr<int64_t>())
        : nullptr;
    launch_sqrelu_deriv_row_rht_persistent_dispatch(
        dh,
        h1_raw,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        sgs,
        amaxes,
        sync_buf,
        data_stochastic_rounding,
        encode_centric,
        scale_stochastic_rounding,
        rng_state_ptr,
        stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_sqrelu_deriv_quantize_row_rht_persistent failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4.view(torch::kFloat4_e2m1fn_x2),
        row_sc.view(torch::kFloat8_e4m3fn),
        col_fp4.view(torch::kFloat4_e2m1fn_x2),
        col_sc.view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 0, 1),
        sgs.narrow(0, 1, 1));
}

template <bool DERIV>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor>
tk_sqrelu_quantize_for_gemm_opt_impl(
    torch::Tensor input,
    torch::Tensor aux,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    if constexpr (DERIV) {
        TORCH_CHECK(aux.is_cuda() && aux.is_contiguous());
        TORCH_CHECK(aux.scalar_type() == torch::kBFloat16 && aux.sizes() == input.sizes());
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    V5QuantSequenceGuard sequence_guard(stream);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);
    TORCH_CHECK(return_transpose, "v5 fused square-ReLU quantizer currently requires return_transpose=True");
    TORCH_CHECK(encode_centric, "v5 fused square-ReLU quantizer currently supports only encode-centric mode");
    TORCH_CHECK(!with_random_sign_mask, "v5 fused square-ReLU quantizer does not fuse random-sign RHT");

    for (char &c : rht_axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    TORCH_CHECK(
        rht_axes == "none" || rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
        rht_axes == "rowcol" || rht_axes == "row_col",
        "Unsupported regular TK square-ReLU RHT axes: ", rht_axes);
    if constexpr (DERIV) {
        TORCH_CHECK(!col_rht,
                    "v5 fused square-ReLU derivative quantizer currently supports row/no-col RHT variants");
    } else {
        TORCH_CHECK(!data_stochastic_rounding && !scale_stochastic_rounding,
                    "v5 fused square-ReLU activation quantizer supports no-SR variants only");
    }

    if constexpr (DERIV) {
        if (row_rht && !col_rht) {
            return tk_sqrelu_deriv_quantize_row_rht_persistent(
                input,
                aux,
                encode_centric,
                data_stochastic_rounding,
                scale_stochastic_rounding,
                rng_seed,
                rng_subsequence);
        }
    } else if (row_rht && !col_rht) {
        return tk_sqrelu_quantize_row_rht_persistent(input, encode_centric);
    }

    auto rng_state = (data_stochastic_rounding || scale_stochastic_rounding)
        ? tk_make_advancing_rng_state(input, rng_seed, rng_subsequence, stream)
        : torch::Tensor();
    const auto *rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const size_t *>(rng_state.data_ptr<int64_t>())
        : nullptr;
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    torch::Tensor row_amax;
    torch::Tensor col_amax;
    if (row_rht && col_rht) {
        auto row_scan = tk_scan_sqrelu_rht16_and_orig_amax(input, aux, true, DERIV, stream);
        auto col_scan = tk_scan_sqrelu_rht16_and_orig_amax(input, aux, false, DERIV, stream);
        row_amax = std::get<0>(row_scan);
        col_amax = std::get<0>(col_scan);
    } else if (row_rht) {
        auto scans = tk_scan_sqrelu_rht16_and_orig_amax(input, aux, true, DERIV, stream);
        row_amax = std::get<0>(scans);
        col_amax = std::get<1>(scans);
    } else if (col_rht) {
        auto scans = tk_scan_sqrelu_rht16_and_orig_amax(input, aux, false, DERIV, stream);
        row_amax = std::get<1>(scans);
        col_amax = std::get<0>(scans);
    } else {
        row_amax = tk_compute_sqrelu_amax_sg(input, aux, DERIV, stream);
        col_amax = row_amax;
    }

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc  = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_u8);
    auto col_sc  = torch::empty({K, scale_stride_t}, opts_u8);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    auto *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    auto *sc_t_ptr = reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr());
    const float *amax_r = reinterpret_cast<const float*>(row_amax.narrow(0, 0, 1).data_ptr());
    const float *amax_c = reinterpret_cast<const float*>(col_amax.narrow(0, 0, 1).data_ptr());
    const IType *aux_ptr = DERIV ? reinterpret_cast<const IType*>(aux.data_ptr()) : nullptr;

    if constexpr (DERIV) {
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            if (row_rht && col_rht) {
                launch_v2_kernel<true, false, true, true, true, true, true, true, false, true>(
                    tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                    scale_stride, scale_stride_t, stream, aux_ptr, rng_state_ptr);
            } else if (row_rht) {
                launch_v2_kernel<true, false, true, true, true, true, false, true, false, true>(
                    tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                    scale_stride, scale_stride_t, stream, aux_ptr, rng_state_ptr);
            } else if (col_rht) {
                launch_v2_kernel<true, false, true, true, true, false, true, true, false, true>(
                    tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                    scale_stride, scale_stride_t, stream, aux_ptr, rng_state_ptr);
            } else {
                launch_v2_kernel<true, false, true, true, true, false, false, true, false, true>(
                    tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                    scale_stride, scale_stride_t, stream, aux_ptr, rng_state_ptr);
            }
        } else if (row_rht && col_rht) {
            launch_v2_kernel<false, false, true, true, false, true, true, true, false, true>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream, aux_ptr);
        } else if (row_rht) {
            launch_v2_kernel<false, false, true, true, false, true, false, true, false, true>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream, aux_ptr);
        } else if (col_rht) {
            launch_v2_kernel<false, false, true, true, false, false, true, true, false, true>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream, aux_ptr);
        } else {
            launch_v2_kernel<false, false, true, true, false, false, false, true, false, true>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream, aux_ptr);
        }
    } else {
        if (row_rht && col_rht) {
            launch_v2_kernel<false, false, true, true, false, true, true, true, true, false>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream);
        } else if (row_rht) {
            launch_v2_kernel<false, false, true, true, false, true, false, true, true, false>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream);
        } else if (col_rht) {
            launch_v2_kernel<false, false, true, true, false, false, true, true, true, false>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream);
        } else {
            launch_v2_kernel<false, false, true, true, false, false, false, true, true, false>(
                tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c, M, K,
                scale_stride, scale_stride_t, stream);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_sqrelu_quantize_for_gemm_opt failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4.view(torch::kFloat4_e2m1fn_x2),
        row_sc.view(torch::kFloat8_e4m3fn),
        col_fp4.view(torch::kFloat4_e2m1fn_x2),
        col_sc.view(torch::kFloat8_e4m3fn),
        row_amax.narrow(0, 1, 1),
        col_amax.narrow(0, 1, 1)
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_sqrelu_quantize_for_gemm_opt(
    torch::Tensor input,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    return tk_sqrelu_quantize_for_gemm_opt_impl<false>(
        input, torch::Tensor(), return_transpose, encode_centric,
        data_stochastic_rounding, scale_stochastic_rounding,
        rht_axes, with_random_sign_mask, rng_seed, rng_subsequence);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_sqrelu_deriv_quantize_for_gemm_opt(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    return tk_sqrelu_quantize_for_gemm_opt_impl<true>(
        dh, h1_raw, return_transpose, encode_centric,
        data_stochastic_rounding, scale_stochastic_rounding,
        rht_axes, with_random_sign_mask, rng_seed, rng_subsequence);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_quantize_row_for_gemm_sr(torch::Tensor input, bool encode_centric) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto amax_buf = torch::empty({3}, opts_f32);
    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = amax_ptr + 1;
    auto *block_count = reinterpret_cast<unsigned int*>(amax_ptr + 2);
    cudaMemsetAsync(amax_ptr, 0, 3 * sizeof(float), stream);

    int64_t n = M * K;
    int amax_blocks = pipelined_amax::grid_size(n);
    int amax_smem = pipelined_amax::smem_size();
    pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        amax_ptr, sg_ptr, block_count, n);

    auto rng_state = tk_make_advancing_rng_state(input, 0, 0, stream);
    auto [row_fp4, row_sc, _col_fp4, _col_sc] =
        tk_quantize_transpose(
            input,
            amax_buf.narrow(0, 0, 1),
            amax_buf.narrow(0, 0, 1),
            false,
            true,
            encode_centric,
            false,
            false,
            false,
            true,
            rng_state);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_quantize_row_for_gemm_sr failed: ", cudaGetErrorString(err));

    if (!encode_centric) {
        TORCH_CHECK(false, "tk_quantize_row_for_gemm_sr currently supports only encode-centric mode");
    }

    return std::make_tuple(
        row_fp4.view(torch::kFloat4_e2m1fn_x2),
        row_sc.view(torch::kFloat8_e4m3fn),
        amax_buf.narrow(0, 1, 1),
        amax_buf.narrow(0, 1, 1)
    );
}


// ═══════════════════════════════════════════════════════════════════
// 3. Grouped dim=0 — fused single-pass with v2 fallback
// ═══════════════════════════════════════════════════════════════════

// Each asynchronous descriptor copy owns its host staging allocation.  The
// caching host allocator event prevents reuse even if the Python keepalive is
// released before the copy completes.
static torch::Tensor allocate_pinned_tma_buffer(size_t n_maps) {
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCPU)
        .pinned_memory(true);
    return torch::empty(
        {static_cast<int64_t>(n_maps * sizeof(CUtensorMap))}, options
    );
}

static void copy_pinned_tma_maps(
    torch::Tensor& device_buffer,
    const torch::Tensor& host_buffer,
    size_t n_maps,
    c10::cuda::CUDAStream stream
) {
    const size_t nbytes = n_maps * sizeof(CUtensorMap);
    v5_check_cuda(
        cudaMemcpyAsync(
            device_buffer.data_ptr(), host_buffer.data_ptr(), nbytes,
            cudaMemcpyHostToDevice, stream
        ),
        "grouped quant TMA descriptor copy failed"
    );
    const auto& data_ptr = host_buffer.storage().data_ptr();
    TORCH_CHECK(
        at::getHostAllocator(at::kCUDA)->record_event(
            host_buffer.data_ptr(), data_ptr.get_context(), stream.unwrap()
        ),
        "grouped quant TMA staging allocation was not owned by the CUDA host allocator"
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0);

    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    TORCH_CHECK(N <= tk_v3::MAX_GROUPS, "Max ", tk_v3::MAX_GROUPS, " splits");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(split_sections[i] % 128 == 0);
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad_tiles = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_fp4);
    auto wc_fp4_col = torch::empty({K, total_rows / 2}, opts_u8);

    auto sg_cat     = torch::empty({N}, opts_f32);
    auto fwd_b_sg   = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);

    auto amax_tensor = torch::empty({N}, opts_f32);
    auto sync_tensor = torch::empty({2 * N}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, N * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * N * sizeof(unsigned int), stream);

    std::vector<torch::Tensor> sc_row_allocs(N);
    std::vector<torch::Tensor> fp4_col_list(N), sc_col_allocs(N);

    tk_v3::FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + N;
    args.num_groups   = N;
    args.sg_output    = sg_cat.data_ptr<float>();
    args.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
    args.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
    args.b_tile_size  = (int)Nb;
    args.total_cols   = (int)K;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    using namespace tk_v3;
    const int blocks_X = K / V3Config::CHUNK_DIM_X;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        args.split_range[i + 1] = args.split_range[i] + (int)M_i;
        args.blocks_per_group[i] = (int)(M_i / V3Config::CHUNK_DIM_Y) * blocks_X;

        sc_row_allocs[i] = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        args.row_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[i].data_ptr());

        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        args.col_data_ptrs[i] = reinterpret_cast<fp4e2m1x2*>(fp4_col_list[i].data_ptr());

        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        sc_col_allocs[i] = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
        args.col_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[i].data_ptr());
        args.col_scale_stride[i] = (int)c_sc_stride;
    }

    const int blocks_Y = total_rows / V3Config::CHUNK_DIM_Y;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int max_concurrent = ci.grp_d0_max_bps * persistent_launch_sms(ci);
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d0_max_bps > 0);

    auto tma_dev_buf = torch::empty({0}, opts_u8);
    auto tma_host_buf = allocate_pinned_tma_buffer(2 * N);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;

        // Create TMA maps in call-owned pinned host memory.
        CUtensorMap* pinned_maps =
            reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            const int64_t ntm_i = M_i / 128;
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(pinned_maps[i], sc_row_allocs[i].data_ptr(), ntm_i, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = K / 128;
            const int64_t sc_col_x_bf16 = (M_i / 64) * 256;
            create_tma_2d(pinned_maps[N + i], sc_col_allocs[i].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * N * sizeof(CUtensorMap))}, opts_u8);
        copy_pinned_tma_maps(tma_dev_buf, tma_host_buf, 2 * N, stream);
        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        fused_group_quantize_kernel_dim0<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr, nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            args);
    } else {
        // ─── v4 persistent grouped (work-stealing, L2 re-reads) ───
        // Need additional sync buffers for persistent barrier
        auto psync_tensor = torch::empty({4}, opts_u32);
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());
        cudaMemsetAsync(psync, 0, 4 * sizeof(unsigned int), stream);

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        int num_persistent = ci.pg_d0_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = N;
        pargs.sg_output    = sg_cat.data_ptr<float>();
        pargs.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
        pargs.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
        pargs.b_tile_size  = (int)Nb;
        pargs.total_cols   = (int)K;
        pargs.swizzle_scales = true;

        for (int i = 0; i <= N; ++i) pargs.split_range[i] = args.split_range[i];
        for (int i = 0; i < N; ++i) {
            pargs.blocks_per_group[i] = args.blocks_per_group[i];
            pargs.row_scale_ptrs[i]   = args.row_scale_ptrs[i];
            pargs.col_data_ptrs[i]    = args.col_data_ptrs[i];
            pargs.col_scale_ptrs[i]   = args.col_scale_ptrs[i];
            pargs.col_scale_stride[i] = args.col_scale_stride[i];
        }

        // Create TMA maps in call-owned pinned host memory.
        CUtensorMap* pinned_maps_p =
            reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            const int64_t ntm_i = M_i / 128;
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            create_tma_2d(pinned_maps_p[i], sc_row_allocs[i].data_ptr(), ntm_i, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = K / 128;
            const int64_t sc_col_x_bf16 = (M_i / 64) * 256;
            create_tma_2d(pinned_maps_p[N + i], sc_col_allocs[i].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * N * sizeof(CUtensorMap))}, opts_u8);
        copy_pinned_tma_maps(tma_dev_buf, tma_host_buf, 2 * N, stream);
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        cudaError_t launch_err;
        if (barrier_free_group_quant_enabled()) {
            launch_err = launch_persistent_group_quantize_two_pass<false>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        } else {
            launch_err = launch_persistent_group_quantize_stream_safe<false>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "persistent grouped dim0 launch failed: ",
                    cudaGetErrorString(launch_err));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_for_gemm failed: ", cudaGetErrorString(err));

    auto wc_sc_row_parts = std::vector<torch::Tensor>(N);
    for (int i = 0; i < N; ++i)
        wc_sc_row_parts[i] = sc_row_allocs[i].view(torch::kFloat8_e4m3fn);
    auto wc_sc_row = torch::cat(wc_sc_row_parts, 0);

    std::vector<torch::Tensor> sc_col_list(N);
    {
        int64_t row_offset = 0;
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            fp4_col_list[i] = wc_fp4_col.narrow(1, row_offset / 2, M_i / 2)
                                         .contiguous().view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[i] = sc_col_allocs[i].view(torch::kFloat8_e4m3fn);
            row_offset += M_i;
        }
    }

    auto mega_buf = tma_host_buf;
    return std::make_tuple(wc_fp4_row, wc_sc_row, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}

// ═══════════════════════════════════════════════════════════════════
// 3b. Grouped dim=0 — NON-PERSISTENT two-pass (multi-stream safe)
//     Pass 1: pipelined grouped amax
//     Pass 2: v2 quantize kernel (pre-computed amax, no barriers)
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_for_gemm_v2(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0);

    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    TORCH_CHECK(N <= tk_v3::MAX_GROUPS, "Max ", tk_v3::MAX_GROUPS, " splits");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(split_sections[i] % 128 == 0);
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad_tiles = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    // Row FP4 (contiguous)
    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_u8).view(torch::kFloat4_e2m1fn_x2);
    // Col FP4 (contiguous — we'll split per-group after)
    auto wc_fp4_col = torch::empty({K, total_rows / 2}, opts_u8);

    // Per-split amax (zeroed for atomicMax)
    auto amaxes = torch::zeros({N}, opts_f32);
    auto sg_cat = torch::empty({N}, opts_f32);
    auto fwd_b_sg  = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);

    // Per-split scale + col allocations
    std::vector<torch::Tensor> sc_row_allocs(N), fp4_col_list(N), sc_col_allocs(N);

    // ── Pass 1: Pipelined grouped amax ──
    {
        std::vector<int64_t> host_offsets(N + 1);
        host_offsets[0] = 0;
        for (int i = 0; i < N; ++i)
            host_offsets[i + 1] = host_offsets[i] + split_sections[i] * K;

        auto d_offsets = torch::empty({N + 1}, torch::dtype(torch::kInt64).device(device));
        cudaMemcpy(
            d_offsets.data_ptr<int64_t>(),
            host_offsets.data(),
            (N + 1) * sizeof(int64_t),
            cudaMemcpyHostToDevice
        );

        int64_t n = total_rows * K;
        int grp_blocks = pipelined_amax::grid_size(n);
        int grp_smem = pipelined_amax::smem_size();
        pipelined_amax::grouped_amax_pipelined_kernel<<<grp_blocks, pipelined_amax::THREADS, grp_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            amaxes.data_ptr<float>(),
            d_offsets.data_ptr<int64_t>(),
            N, n);
    }

    // sg = amax / 2688
    compute_sg_kernel<<<1, N, 0, stream>>>(
        amaxes.data_ptr<float>(), sg_cat.data_ptr<float>(), N);

    // ── Build v2 kernel args ──
    grp_kernel::MultiAmaxCastTransposeFusionArgs kernel_args;
    memset(&kernel_args, 0, sizeof(kernel_args));
    kernel_args.num_tensors = N;
    kernel_args.swizzle_scales = true;
    kernel_args.sg_output = sg_cat.data_ptr<float>();
    kernel_args.fwd_b_sg = fwd_b_sg.data_ptr<float>();
    kernel_args.dgrad_b_sg = dgrad_b_sg.data_ptr<float>();
    kernel_args.b_tile_size = (int)Nb;
    kernel_args.total_cols = (int)K;
    kernel_args.split_sections_range[0] = 0;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        kernel_args.split_sections_range[i + 1] = kernel_args.split_sections_range[i] + (int)M_i;
        kernel_args.rowwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.colwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);

        sc_row_allocs[i] = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        kernel_args.output_rowwise_scale_inv_list[i] = sc_row_allocs[i].data_ptr();

        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        kernel_args.output_colwise_data_list[i] = fp4_col_list[i].data_ptr();

        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        sc_col_allocs[i] = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
        kernel_args.output_colwise_scale_inv_list[i] = sc_col_allocs[i].data_ptr();
        kernel_args.output_colwise_scale_stride[i] = (int)c_sc_stride;
    }

    // ── Pass 2: v2 grouped quantize (non-persistent) ──
    alignas(64) CUtensorMap tmap_in{}, tmap_out{};
    create_tma_2d(tmap_in, input.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 4);

    nvfp4_scale_t* scales_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());
    launch_group_kernel_v2<false, true>(
        tmap_in, tmap_in, tmap_out, scales_ptr,
        total_rows, K, scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_for_gemm_v2 failed: ", cudaGetErrorString(err));

    // ── Reshape outputs (same as v5) ──
    auto wc_sc_row_parts = std::vector<torch::Tensor>(N);
    for (int i = 0; i < N; ++i)
        wc_sc_row_parts[i] = sc_row_allocs[i].view(torch::kFloat8_e4m3fn);
    auto wc_sc_row = torch::cat(wc_sc_row_parts, 0);

    std::vector<torch::Tensor> sc_col_list(N);
    {
        int64_t row_offset = 0;
        for (int i = 0; i < N; ++i) {
            const int64_t M_i = split_sections[i];
            fp4_col_list[i] = fp4_col_list[i].view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[i] = sc_col_allocs[i].view(torch::kFloat8_e4m3fn);
            row_offset += M_i;
        }
    }

    auto mega_buf = torch::empty({0}, opts_u8);
    return std::make_tuple(wc_fp4_row, wc_sc_row, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_split_for_gemm_v2(
    torch::Tensor w1,
    torch::Tensor w3
) {
    TORCH_CHECK(w1.is_cuda() && w1.is_contiguous());
    TORCH_CHECK(w3.is_cuda() && w3.is_contiguous());
    TORCH_CHECK(w1.scalar_type() == torch::kBFloat16 && w1.dim() == 2);
    TORCH_CHECK(w3.scalar_type() == torch::kBFloat16 && w3.dim() == 2);
    TORCH_CHECK(w1.sizes() == w3.sizes(), "w1 and w3 must have identical shapes");
    TORCH_CHECK(w1.size(0) % 128 == 0 && w1.size(1) % 128 == 0);

    const int64_t H = w1.size(0);
    const int64_t K = w1.size(1);
    const int64_t total_rows = 2 * H;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = w1.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    const int64_t total_fwd_tiles = total_rows / 256;
    const int64_t total_dgrad_tiles = 2 * (K / 256);

    auto row_fp4 = torch::empty({total_rows, K / 2}, opts_u8).view(torch::kFloat4_e2m1fn_x2);
    auto amaxes = torch::zeros({2}, opts_f32);
    auto sg_cat = torch::empty({2}, opts_f32);
    auto fwd_b_sg = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad_tiles}, opts_f32);
    // The final two words are operation-local completion counters for the
    // two independent amax launches.
    auto tmp_sg = torch::zeros({4}, opts_f32);

    std::vector<torch::Tensor> sc_row_allocs(2), fp4_col_list(2), sc_col_allocs(2);
    for (int i = 0; i < 2; ++i) {
        sc_row_allocs[i] = torch::empty({H / 128, ntk_r, 512}, opts_u8);
        fp4_col_list[i] = torch::empty({K, H / 2}, opts_u8);
        sc_col_allocs[i] = torch::zeros({ntm_c, H / 64, 512}, opts_u8);
    }

    const int64_t n = H * K;
    const int amax_blocks = pipelined_amax::grid_size(n);
    const int amax_smem = pipelined_amax::smem_size();
    cudaMemsetAsync(amaxes.data_ptr<float>(), 0, 2 * sizeof(float), stream);
    pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w1.data_ptr()),
        amaxes.data_ptr<float>() + 0,
        tmp_sg.data_ptr<float>() + 0,
        reinterpret_cast<unsigned int*>(tmp_sg.data_ptr<float>() + 2) + 0,
        n);
    pipelined_amax::fused_amax_pipelined_kernel<<<amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w3.data_ptr()),
        amaxes.data_ptr<float>() + 1,
        tmp_sg.data_ptr<float>() + 1,
        reinterpret_cast<unsigned int*>(tmp_sg.data_ptr<float>() + 2) + 1,
        n);
    compute_sg_kernel<<<1, 2, 0, stream>>>(amaxes.data_ptr<float>(), sg_cat.data_ptr<float>(), 2);

    grp_kernel::MultiAmaxCastTransposeFusionArgs kernel_args;
    memset(&kernel_args, 0, sizeof(kernel_args));
    kernel_args.num_tensors = 2;
    kernel_args.num_input_tensor_maps = 2;
    kernel_args.swizzle_scales = true;
    kernel_args.sg_output = sg_cat.data_ptr<float>();
    kernel_args.fwd_b_sg = fwd_b_sg.data_ptr<float>();
    kernel_args.dgrad_b_sg = dgrad_b_sg.data_ptr<float>();
    kernel_args.b_tile_size = 256;
    kernel_args.total_cols = (int)K;
    kernel_args.split_sections_range[0] = 0;
    kernel_args.split_sections_range[1] = (int)H;
    kernel_args.split_sections_range[2] = (int)(2 * H);
    kernel_args.rowwise_amax_list[0] = amaxes.data_ptr<float>() + 0;
    kernel_args.rowwise_amax_list[1] = amaxes.data_ptr<float>() + 1;
    kernel_args.colwise_amax_list[0] = amaxes.data_ptr<float>() + 0;
    kernel_args.colwise_amax_list[1] = amaxes.data_ptr<float>() + 1;
    kernel_args.output_rowwise_scale_inv_list[0] = sc_row_allocs[0].data_ptr();
    kernel_args.output_rowwise_scale_inv_list[1] = sc_row_allocs[1].data_ptr();
    kernel_args.output_colwise_data_list[0] = fp4_col_list[0].data_ptr();
    kernel_args.output_colwise_data_list[1] = fp4_col_list[1].data_ptr();
    kernel_args.output_colwise_scale_inv_list[0] = sc_col_allocs[0].data_ptr();
    kernel_args.output_colwise_scale_inv_list[1] = sc_col_allocs[1].data_ptr();
    kernel_args.output_colwise_scale_stride[0] = (int)(((H / 16) + 3) / 4 * 4);
    kernel_args.output_colwise_scale_stride[1] = (int)(((H / 16) + 3) / 4 * 4);

    alignas(64) CUtensorMap tmap_w1{}, tmap_w3{}, tmap_out{};
    create_tma_2d(tmap_w1, w1.data_ptr(), H, K, grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_w3, w3.data_ptr(), H, K, grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 4);
    auto* scales_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[0].data_ptr());
    launch_group_kernel_v2<false, true>(
        tmap_w1, tmap_w3, tmap_out, scales_ptr,
        total_rows, K, scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_split_for_gemm_v2 failed: ",
                cudaGetErrorString(err));

    auto row_sc = torch::cat({
        sc_row_allocs[0].view(torch::kFloat8_e4m3fn),
        sc_row_allocs[1].view(torch::kFloat8_e4m3fn)
    }, 0);

    std::vector<torch::Tensor> sc_col_list = {
        sc_col_allocs[0].view(torch::kFloat8_e4m3fn),
        sc_col_allocs[1].view(torch::kFloat8_e4m3fn),
    };
    fp4_col_list[0] = fp4_col_list[0].view(torch::kFloat4_e2m1fn_x2);
    fp4_col_list[1] = fp4_col_list[1].view(torch::kFloat4_e2m1fn_x2);

    auto mega_buf = torch::empty({0}, opts_u8);
    return std::make_tuple(row_fp4, row_sc, fwd_b_sg,
                           fp4_col_list, sc_col_list, dgrad_b_sg, sg_cat, mega_buf);
}


// ═══════════════════════════════════════════════════════════════════
// 3c. Multi-stream graph-safe split API:
//     - v2_alloc: pre-allocate all output tensors (call on capture/main stream)
//     - v2_launch: kernel-only dispatch (safe on ANY stream during graph capture)
// ═══════════════════════════════════════════════════════════════════

// Returns: (wc_fp4_row, amaxes, sg_cat, fwd_b_sg, dgrad_b_sg, d_offsets,
//           sc_row_list, fp4_col_list, sc_col_list)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
tk_group_quantize_v2_alloc(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    // Pre-allocate everything on the CURRENT stream (capture stream)
    auto wc_fp4_row = torch::empty({total_rows, K / 2}, opts_u8);
    auto amaxes     = torch::zeros({N}, opts_f32);
    auto sg_cat     = torch::empty({N}, opts_f32);
    auto fwd_b_sg   = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg = torch::empty({total_dgrad}, opts_f32);
    auto d_offsets  = torch::empty({N + 1}, torch::dtype(torch::kInt64).device(device));

    std::vector<torch::Tensor> sc_row_list(N), fp4_col_list(N), sc_col_list(N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        sc_row_list[i]  = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        sc_col_list[i]  = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
    }

    return std::make_tuple(wc_fp4_row, amaxes, sg_cat, fwd_b_sg, dgrad_b_sg,
                           d_offsets, sc_row_list, fp4_col_list, sc_col_list);
}

// Kernel-only dispatch — NO allocations, safe on any stream during graph capture.
// Takes pre-allocated buffers from v2_alloc.
// Returns: (fp4_row_viewed, fwd_b_sg, dgrad_b_sg, sg_cat)
// Caller reconstructs wc_sc_row = torch.cat([sc_row[i].view(torch.kFloat8_e4m3fn) for i in range(N)])
static void launch_group_quantize_v2_into_outputs(
    torch::Tensor input,
    const std::vector<int64_t>& split_sections,
    torch::Tensor wc_fp4_row,
    torch::Tensor amaxes,
    torch::Tensor sg_cat,
    torch::Tensor fwd_b_sg,
    torch::Tensor dgrad_b_sg,
    torch::Tensor d_offsets,
    const std::vector<torch::Tensor>& sc_row_list,
    const std::vector<torch::Tensor>& fp4_col_list,
    const std::vector<torch::Tensor>& sc_col_list
) {
    auto& ci = get_cached_info();
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    // ── Pass 1: Pipelined grouped amax (cudaMemcpyAsync from pinned is graph-safe) ──
    {
        cudaMemsetAsync(amaxes.data_ptr<float>(), 0, N * sizeof(float), stream);

        std::vector<int64_t> host_offsets(N + 1);
        host_offsets[0] = 0;
        for (int i = 0; i < N; ++i)
            host_offsets[i + 1] = host_offsets[i] + split_sections[i] * K;

        cudaMemcpy(
            d_offsets.data_ptr<int64_t>(),
            host_offsets.data(),
            (N + 1) * sizeof(int64_t),
            cudaMemcpyHostToDevice
        );

        int64_t n = total_rows * K;
        int grp_blocks = pipelined_amax::grid_size(n);
        int grp_smem = pipelined_amax::smem_size();
        pipelined_amax::grouped_amax_pipelined_kernel<<<grp_blocks, pipelined_amax::THREADS, grp_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            amaxes.data_ptr<float>(),
            d_offsets.data_ptr<int64_t>(),
            N, n);
    }

    // sg = amax / 2688
    compute_sg_kernel<<<1, N, 0, stream>>>(
        amaxes.data_ptr<float>(), sg_cat.data_ptr<float>(), N);

    // ── Build kernel args ──
    grp_kernel::MultiAmaxCastTransposeFusionArgs kernel_args;
    memset(&kernel_args, 0, sizeof(kernel_args));
    kernel_args.num_tensors = N;
    kernel_args.swizzle_scales = true;
    kernel_args.sg_output = sg_cat.data_ptr<float>();
    kernel_args.fwd_b_sg = fwd_b_sg.data_ptr<float>();
    kernel_args.dgrad_b_sg = dgrad_b_sg.data_ptr<float>();
    kernel_args.b_tile_size = (int)Nb;
    kernel_args.total_cols = (int)K;
    kernel_args.split_sections_range[0] = 0;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        kernel_args.split_sections_range[i + 1] = kernel_args.split_sections_range[i] + (int)M_i;
        kernel_args.rowwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.colwise_amax_list[i] = (void*)(amaxes.data_ptr<float>() + i);
        kernel_args.output_rowwise_scale_inv_list[i] = sc_row_list[i].data_ptr();
        kernel_args.output_colwise_data_list[i] = fp4_col_list[i].data_ptr();
        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        kernel_args.output_colwise_scale_inv_list[i] = sc_col_list[i].data_ptr();
        kernel_args.output_colwise_scale_stride[i] = (int)c_sc_stride;
    }

    // ── Pass 2: v2 grouped quantize (non-persistent, no allocations) ──
    alignas(64) CUtensorMap tmap_in{}, tmap_out{};
    create_tma_2d(tmap_in, input.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K,
                  grp_kernel::BUFF_DIM_Y, grp_kernel::BUFF_DIM_X, K, 4);

    nvfp4_scale_t* scales_ptr = reinterpret_cast<nvfp4_scale_t*>(sc_row_list[0].data_ptr());
    launch_group_kernel_v2<false, true>(
        tmap_in, tmap_in, tmap_out, scales_ptr,
        total_rows, K, scale_stride,
        kernel_args, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch_group_quantize_v2_into_outputs failed: ", cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_v2_launch(
    torch::Tensor input,
    std::vector<int64_t> split_sections,
    // Pre-allocated buffers from v2_alloc:
    torch::Tensor wc_fp4_row,
    torch::Tensor amaxes,
    torch::Tensor sg_cat,
    torch::Tensor fwd_b_sg,
    torch::Tensor dgrad_b_sg,
    torch::Tensor d_offsets,
    std::vector<torch::Tensor> sc_row_list,
    std::vector<torch::Tensor> fp4_col_list,
    std::vector<torch::Tensor> sc_col_list
) {
    launch_group_quantize_v2_into_outputs(
        input, split_sections, wc_fp4_row, amaxes, sg_cat, fwd_b_sg, dgrad_b_sg,
        d_offsets, sc_row_list, fp4_col_list, sc_col_list);

    auto wc_fp4_view = wc_fp4_row.view(torch::kFloat4_e2m1fn_x2);
    return std::make_tuple(wc_fp4_view, fwd_b_sg, dgrad_b_sg, sg_cat);
}


// ═══════════════════════════════════════════════════════════════════
// 3d. V5 Persistent/Fused split API for multi-stream graph capture:
//     - v5_alloc: pre-allocate all output tensors (call on capture/main stream)
//     - v5_launch: kernel-only dispatch (safe on ANY stream during graph capture)
// ═══════════════════════════════════════════════════════════════════

// Returns: (wc_fp4_row, wc_fp4_col, sg_cat, fwd_b_sg, dgrad_b_sg,
//           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
//           sc_row_list, fp4_col_list, sc_col_list)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
tk_group_quantize_v5_alloc(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "grouped v5 alloc expects a 2D BF16 input");
    c10::cuda::CUDAGuard device_guard(input.device());
    const int64_t total_rows = input.size(0), K = input.size(1);
    const int N = (int)split_sections.size();
    TORCH_CHECK(N > 0 && N <= tk_v3::MAX_GROUPS,
                "grouped v5 alloc split count must be in [1, ", tk_v3::MAX_GROUPS, "]");
    TORCH_CHECK(total_rows % 128 == 0 && K % 128 == 0,
                "grouped v5 alloc dimensions must be multiples of 128");
    const auto device = input.device();
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;

    int64_t sum_splits = 0, total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i) {
        TORCH_CHECK(split_sections[i] > 0 && split_sections[i] % 128 == 0,
                    "grouped v5 alloc split sections must be positive multiples of 128");
        sum_splits += split_sections[i];
        total_fwd_tiles += split_sections[i] / Nb;
    }
    TORCH_CHECK(sum_splits == total_rows);
    const int64_t dgrad_tiles_per = K / Nb;
    const int64_t total_dgrad = (int64_t)N * dgrad_tiles_per;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);

    // Pre-allocate everything on the CURRENT stream (capture stream)
    auto wc_fp4_row   = torch::empty({total_rows, K / 2}, opts_fp4);
    auto wc_fp4_col   = torch::empty({K, total_rows / 2}, opts_u8);
    auto sg_cat       = torch::empty({N}, opts_f32);
    auto fwd_b_sg     = torch::empty({total_fwd_tiles}, opts_f32);
    auto dgrad_b_sg   = torch::empty({total_dgrad}, opts_f32);
    auto amax_tensor  = torch::empty({N}, opts_f32);
    auto sync_tensor  = torch::empty({2 * N}, opts_u32);
    auto psync_tensor = torch::empty({4}, opts_u32);
    auto tma_dev_buf  = torch::empty({(int64_t)(2 * N * (int64_t)sizeof(CUtensorMap))}, opts_u8);

    std::vector<torch::Tensor> sc_row_list(N), fp4_col_list(N), sc_col_list(N);
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        sc_row_list[i]  = torch::empty({M_i / 128, ntk_r, 512}, opts_u8);
        fp4_col_list[i] = torch::empty({K, M_i / 2}, opts_u8);
        sc_col_list[i]  = torch::zeros({ntm_c, M_i / 64, 512}, opts_u8);
    }

    auto tma_host_buf = torch::empty(
        {(int64_t)(2 * N * sizeof(CUtensorMap))},
        torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    CUtensorMap* pinned_maps = reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        const int64_t ntm_i = M_i / 128;
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(pinned_maps[i], sc_row_list[i].data_ptr(), ntm_i,
                      sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        const int64_t sc_col_x_bf16 = (M_i / 64) * 256;
        create_tma_2d(pinned_maps[N + i], sc_col_list[i].data_ptr(), ntm_c,
                      sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
    v5_check_cuda(cudaMemcpy(
        tma_dev_buf.data_ptr(), pinned_maps,
        2 * N * sizeof(CUtensorMap), cudaMemcpyHostToDevice),
        "tk_group_quantize_v5_alloc descriptor copy failed");

    return std::make_tuple(wc_fp4_row, wc_fp4_col, sg_cat, fwd_b_sg, dgrad_b_sg,
                           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
                           sc_row_list, fp4_col_list, sc_col_list);
}

// Kernel-only dispatch — NO allocations, safe on any stream during graph capture.
// Takes pre-allocated buffers from v5_alloc.
// Returns: (fp4_row_viewed, fwd_b_sg, dgrad_b_sg, sg_cat) — same as v2_launch
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_group_quantize_v5_launch(
    torch::Tensor input,
    std::vector<int64_t> split_sections,
    // Pre-allocated buffers from v5_alloc:
    torch::Tensor wc_fp4_row,
    torch::Tensor wc_fp4_col,
    torch::Tensor sg_cat,
    torch::Tensor fwd_b_sg,
    torch::Tensor dgrad_b_sg,
    torch::Tensor amax_tensor,
    torch::Tensor sync_tensor,
    torch::Tensor psync_tensor,
    torch::Tensor tma_dev_buf,
    std::vector<torch::Tensor> sc_row_list,
    std::vector<torch::Tensor> fp4_col_list,
    std::vector<torch::Tensor> sc_col_list
) {
    TORCH_CHECK(input.is_cuda(), "grouped v5 launch input must be CUDA");
    const auto device = input.device();
    v5_require_tensor_device(wc_fp4_row, device, "wc_fp4_row");
    v5_require_tensor_device(wc_fp4_col, device, "wc_fp4_col");
    v5_require_tensor_device(sg_cat, device, "sg_cat");
    v5_require_tensor_device(fwd_b_sg, device, "fwd_b_sg");
    v5_require_tensor_device(dgrad_b_sg, device, "dgrad_b_sg");
    v5_require_tensor_device(amax_tensor, device, "amax_tensor");
    v5_require_tensor_device(sync_tensor, device, "sync_tensor");
    v5_require_tensor_device(psync_tensor, device, "psync_tensor");
    v5_require_tensor_device(tma_dev_buf, device, "tma_dev_buf");
    v5_require_tensor_vector_device(sc_row_list, device, "sc_row_list");
    v5_require_tensor_vector_device(fp4_col_list, device, "fp4_col_list");
    v5_require_tensor_vector_device(sc_col_list, device, "sc_col_list");
    const int N = (int)split_sections.size();
    TORCH_CHECK(N > 0 && N <= tk_v3::MAX_GROUPS,
                "grouped v5 launch split count must be in [1, ", tk_v3::MAX_GROUPS, "]");
    TORCH_CHECK(sc_row_list.size() == static_cast<size_t>(N) &&
                fp4_col_list.size() == static_cast<size_t>(N) &&
                sc_col_list.size() == static_cast<size_t>(N),
                "grouped v5 launch output vector lengths must match split_sections");
    const c10::cuda::CUDAGuard device_guard(device);
    const auto& ci = get_cached_info();
    const int64_t total_rows = input.size(0), K = input.size(1);
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    V5QuantSequenceGuard sequence_guard(stream);
    constexpr int64_t Nb = 256;
    const int64_t ntk_r = K / 64;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    // Zero sync buffers (cudaMemsetAsync is graph-safe)
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, N * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * N * sizeof(unsigned int), stream);
    cudaMemsetAsync(psync_tensor.data_ptr(), 0, 4 * sizeof(unsigned int), stream);

    int64_t total_fwd_tiles = 0;
    for (int i = 0; i < N; ++i)
        total_fwd_tiles += split_sections[i] / Nb;

    // ── Build FusedGroupArgs ──
    tk_v3::FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + N;
    args.num_groups   = N;
    args.sg_output    = sg_cat.data_ptr<float>();
    args.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
    args.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
    args.b_tile_size  = (int)Nb;
    args.total_cols   = (int)K;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    using namespace tk_v3;
    const int blocks_X = K / V3Config::CHUNK_DIM_X;

    for (int i = 0; i < N; ++i) {
        const int64_t M_i = split_sections[i];
        args.split_range[i + 1] = args.split_range[i] + (int)M_i;
        args.blocks_per_group[i] = (int)(M_i / V3Config::CHUNK_DIM_Y) * blocks_X;
        args.row_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_row_list[i].data_ptr());
        args.col_data_ptrs[i] = reinterpret_cast<fp4e2m1x2*>(fp4_col_list[i].data_ptr());
        const int64_t c_sc_stride = ((M_i / 16) + 3) / 4 * 4;
        args.col_scale_ptrs[i] = reinterpret_cast<nvfp4_scale_t*>(sc_col_list[i].data_ptr());
        args.col_scale_stride[i] = (int)c_sc_stride;
    }

    const int blocks_Y = total_rows / V3Config::CHUNK_DIM_Y;
    const int total_grid = blocks_X * blocks_Y;
    const int max_concurrent = ci.grp_d0_max_bps * persistent_launch_sms(ci);
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d0_max_bps > 0);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;
        launch_manual_barrier_kernel_capture_safe(
            "fused grouped dim0 quantizer", fused_group_quantize_kernel_dim0<true>,
            grid, dim3(V3_THREADS), dshmem, stream,
            tmap_in, tmap_out, tmap_out_t,
            nullptr, nullptr,
            total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
            args);
    } else {
        // ─── Persistent grouped path ───
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, wc_fp4_row.data_ptr(), total_rows, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
        create_tma_2d(tmap_out_t, wc_fp4_col.data_ptr(), K, total_rows, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, total_rows, 4);

        int num_persistent = ci.pg_d0_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = N;
        pargs.sg_output    = sg_cat.data_ptr<float>();
        pargs.fwd_b_sg     = fwd_b_sg.data_ptr<float>();
        pargs.dgrad_b_sg   = dgrad_b_sg.data_ptr<float>();
        pargs.b_tile_size  = (int)Nb;
        pargs.total_cols   = (int)K;
        pargs.swizzle_scales = true;

        for (int i = 0; i <= N; ++i) pargs.split_range[i] = args.split_range[i];
        for (int i = 0; i < N; ++i) {
            pargs.blocks_per_group[i] = args.blocks_per_group[i];
            pargs.row_scale_ptrs[i]   = args.row_scale_ptrs[i];
            pargs.col_data_ptrs[i]    = args.col_data_ptrs[i];
            pargs.col_scale_ptrs[i]   = args.col_scale_ptrs[i];
            pargs.col_scale_stride[i] = args.col_scale_stride[i];
        }
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        cudaError_t launch_err;
        if (barrier_free_group_quant_enabled()) {
            launch_err = launch_persistent_group_quantize_two_pass<false>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        } else {
            launch_err = launch_persistent_group_quantize_stream_safe<false>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                total_rows, K, scale_stride, ((total_rows / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "stream-safe persistent grouped dim0 launch failed: ",
                    cudaGetErrorString(launch_err));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_v5_launch failed: ", cudaGetErrorString(err));

    // ── Only view (zero-allocation) — caller does torch::cat on capture stream ──
    auto wc_fp4_view = wc_fp4_row.view(torch::kFloat4_e2m1fn_x2);
    return std::make_tuple(wc_fp4_view, fwd_b_sg, dgrad_b_sg, sg_cat);
}

struct TkDataSRAxes {
    bool row;
    bool col;
};

static TkDataSRAxes resolve_data_sr_axes(
    bool enabled,
    std::string axes,
    const char* context
) {
    for (char &c : axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    if (axes == "all" || axes == "rowcol" || axes == "row_col") {
        axes = "both";
    } else if (axes == "column" || axes == "columns" || axes == "wgrad") {
        axes = "col";
    } else if (axes == "dgrad") {
        axes = "row";
    } else if (axes == "off" || axes == "0") {
        axes = "none";
    }
    TORCH_CHECK(
        axes == "none" || axes == "row" || axes == "col" || axes == "both",
        "Unsupported ", context, " data SR axes: ", axes);
    return {
        enabled && (axes == "row" || axes == "both"),
        enabled && (axes == "col" || axes == "both"),
    };
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_group_quantize_dim1_for_gemm(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);

    const int64_t M = input.size(0), N_total = input.size(1);
    TORCH_CHECK(M % 128 == 0 && N_total % 128 == 0);
    const int G = (int)col_split_sections.size();
    TORCH_CHECK(G <= tk_v3::MAX_GROUPS, "Max ", tk_v3::MAX_GROUPS, " groups");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);

    int64_t sum_cols = 0;
    for (int i = 0; i < G; ++i) {
        TORCH_CHECK(col_split_sections[i] % 128 == 0);
        sum_cols += col_split_sections[i];
    }
    TORCH_CHECK(sum_cols == N_total);

    auto amax_tensor = torch::empty({G}, opts_f32);
    auto sync_tensor = torch::empty({2 * G}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);

    auto sg_per_group = torch::empty({G}, opts_f32);
    auto fp4_row_full = torch::empty({M, N_total / 2}, opts_u8);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    using namespace tk_v3;
    FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + G;
    args.num_groups   = G;
    args.sg_output    = sg_per_group.data_ptr<float>();
    args.fwd_b_sg     = nullptr;
    args.dgrad_b_sg   = nullptr;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;

    std::vector<torch::Tensor> sc_row_allocs(G), fp4_col_allocs(G), sc_col_allocs(G);

    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        args.split_range[g + 1] = args.split_range[g] + (int)N_g;
        args.blocks_per_group[g] = (int)(N_g / V3Config::CHUNK_DIM_X) * blocks_Y;

        const int64_t ntk_r_g = N_g / 64;
        sc_row_allocs[g] = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
        args.row_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());

        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        args.col_data_ptrs[g] = reinterpret_cast<fp4e2m1x2*>(fp4_col_allocs[g].data_ptr());

        const int64_t ntm_c_g = N_g / 128;
        const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
        sc_col_allocs[g] = torch::zeros({ntm_c_g, ntk_c, 512}, opts_u8);
        args.col_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)col_sc_stride;
    }

    auto fp4_col_full = torch::empty({N_total, M / 2}, opts_u8);

    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int max_concurrent = ci.grp_d1_max_bps * persistent_launch_sms(ci);
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d1_max_bps > 0);

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;
    auto tma_dev_buf = torch::empty({0}, opts_u8);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;

        // Create TMA maps on host, copy to device tensor
        alignas(64) CUtensorMap host_tma_maps[2 * tk_v3::MAX_GROUPS];
        for (int g = 0; g < G; ++g) {
            const int64_t N_g = col_split_sections[g];
            const int64_t ntk_r_g = N_g / 64;
            const int64_t sc_row_x_bf16 = ntk_r_g * 256;
            create_tma_2d(host_tma_maps[g], sc_row_allocs[g].data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = N_g / 128;
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(host_tma_maps[G + g], sc_col_allocs[g].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * G * sizeof(CUtensorMap))}, opts_u8);
        cudaMemcpyAsync(tma_dev_buf.data_ptr(), host_tma_maps, 2 * G * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        fused_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            args);
    } else {
        // ─── v4 persistent grouped (work-stealing, L2 re-reads) ───
        auto psync_tensor = torch::empty({4}, opts_u32);
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());
        cudaMemsetAsync(psync, 0, 4 * sizeof(unsigned int), stream);

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        int num_persistent = ci.pg_d1_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = G;
        pargs.sg_output    = sg_per_group.data_ptr<float>();
        pargs.fwd_b_sg     = nullptr;
        pargs.dgrad_b_sg   = nullptr;
        pargs.swizzle_scales = true;

        for (int g = 0; g <= G; ++g) pargs.split_range[g] = args.split_range[g];
        for (int g = 0; g < G; ++g) {
            pargs.blocks_per_group[g] = args.blocks_per_group[g];
            pargs.row_scale_ptrs[g]   = args.row_scale_ptrs[g];
            pargs.col_data_ptrs[g]    = args.col_data_ptrs[g];
            pargs.col_scale_ptrs[g]   = args.col_scale_ptrs[g];
            pargs.col_scale_stride[g] = args.col_scale_stride[g];
        }

        // Create TMA maps on host, copy to device tensor
        alignas(64) CUtensorMap host_tma_maps[2 * tk_v3::MAX_GROUPS];
        for (int g = 0; g < G; ++g) {
            const int64_t N_g = col_split_sections[g];
            const int64_t ntk_r_g = N_g / 64;
            const int64_t sc_row_x_bf16 = ntk_r_g * 256;
            create_tma_2d(host_tma_maps[g], sc_row_allocs[g].data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

            const int64_t ntm_c_g = N_g / 128;
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(host_tma_maps[G + g], sc_col_allocs[g].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        }
        tma_dev_buf = torch::empty({(int64_t)(2 * G * sizeof(CUtensorMap))}, opts_u8);
        cudaMemcpyAsync(tma_dev_buf.data_ptr(), host_tma_maps, 2 * G * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        cudaError_t launch_err;
        if (barrier_free_group_quant_enabled()) {
            launch_err = launch_persistent_group_quantize_two_pass<true>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        } else {
            launch_err = launch_persistent_group_quantize_stream_safe<true>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "persistent grouped dim1 launch failed: ",
                    cudaGetErrorString(launch_err));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_for_gemm failed: ", cudaGetErrorString(err));

    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    std::vector<torch::Tensor> sc_row_u8_list(G), sc_col_u8_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];

        // narrow along dim=1 is non-contiguous — return as uint8 view (caller does .contiguous().view(fp4))
        fp4_row_list[g] = fp4_row_full.narrow(1, col_offset / 2, N_g / 2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_row_u8_list[g] = sc_row_allocs[g];

        fp4_col_list[g] = fp4_col_full.narrow(0, col_offset, N_g).view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_col_u8_list[g] = sc_col_allocs[g];
        col_offset += N_g;
    }

    // Concatenate row/col scales in C++ — avoids Python torch.cat round-trip
    auto sc_row_cat = torch::cat(sc_row_u8_list, 1).view(torch::kFloat8_e4m3fn);
    auto sc_col_cat = torch::cat(sc_col_u8_list, 0).view(torch::kFloat8_e4m3fn);
    auto fp4_row_view = fp4_row_full.view(torch::kFloat4_e2m1fn_x2);
    auto fp4_col_view = fp4_col_full.view(torch::kFloat4_e2m1fn_x2);

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list,
                           fp4_row_view, sc_row_cat,
                           fp4_col_view, sc_col_cat);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_group_quantize_dim1_split3_for_gemm(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    bool data_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    std::string data_sr_axes
) {
    std::array<torch::Tensor, 3> inputs{input0, input1, input2};
    for (const auto& input : inputs) {
        TORCH_CHECK(input.is_cuda());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.stride(1) == 1,
                    "split3 inputs must be contiguous along the inner dimension");
    }
    TORCH_CHECK(input1.device() == input0.device() && input2.device() == input0.device(),
                "split3 inputs must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(input0.device());
    const auto set_device_err = cudaSetDevice(input0.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before v5 split3 quantization: ",
                cudaGetErrorString(set_device_err));
    const int64_t M = input0.size(0);
    TORCH_CHECK(input1.size(0) == M && input2.size(0) == M,
                "split3 inputs must share the same row count");
    const int64_t n0 = input0.size(1), n1 = input1.size(1), n2 = input2.size(1);
    const int64_t N_total = n0 + n1 + n2;
    TORCH_CHECK(M % 128 == 0 && N_total % 128 == 0);
    TORCH_CHECK(n0 % 128 == 0 && n1 % 128 == 0 && n2 % 128 == 0);
    constexpr int G = 3;

    auto stream = at::cuda::getCurrentCUDAStream();
    V5QuantSequenceGuard sequence_guard(stream);
    auto device = input0.device();
    const auto sr_axes = resolve_data_sr_axes(
        data_stochastic_rounding, std::move(data_sr_axes), "split3");
    const bool any_data_sr = sr_axes.row || sr_axes.col;
    const bool use_two_pass_sr =
        sr_axes.row && sr_axes.col && split3_two_pass_sr_enabled();
    auto rng_state = any_data_sr
        ? tk_make_advancing_rng_state(
              input0, rng_seed, rng_subsequence, stream,
              use_two_pass_sr ? G : 1)
        : torch::empty(
              {0}, torch::dtype(torch::kInt64).device(input0.device()));
    const auto* rng_state_ptr = rng_state.numel() != 0
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    auto amax_tensor = torch::empty({G}, opts_f32);
    auto sync_tensor = torch::empty({2 * G}, opts_u32);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);

    auto sg_per_group = torch::empty({G}, opts_f32);
    auto fp4_row_full = torch::empty({M, N_total / 2}, opts_u8);
    auto fp4_col_full = torch::empty({N_total, M / 2}, opts_u8);

    if (use_two_pass_sr) {
        std::array<int64_t, G> splits{n0, n1, n2};
        std::vector<torch::Tensor> sc_row_allocs(G), sc_col_allocs(G);
        const int64_t ntm_r = M / 128;
        const int64_t ntk_c = M / 64;

        for (int g = 0; g < G; ++g) {
            const int64_t N_g = splits[g];
            const int64_t ntk_r_g = N_g / 64;
            const int64_t ntm_c_g = N_g / 128;
            sc_row_allocs[g] = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
            sc_col_allocs[g] = torch::empty({ntm_c_g, ntk_c, 512}, opts_u8);

            const int64_t n = inputs[g].numel();
            const int amax_blocks = pipelined_amax::grid_size(n);
            const int amax_smem = pipelined_amax::smem_size();
            pipelined_amax::fused_amax_pipelined_kernel<<<
                amax_blocks, pipelined_amax::THREADS, amax_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(inputs[g].data_ptr()),
                amax_tensor.data_ptr<float>() + g,
                sg_per_group.data_ptr<float>() + g,
                reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>()) + g,
                n);
        }

        int64_t col_offset = 0;
        for (int g = 0; g < G; ++g) {
            const int64_t N_g = splits[g];
            const int64_t scale_stride = ((N_g / 16) + 3) / 4 * 4;
            const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

            auto *row_base =
                fp4_row_full.data_ptr<uint8_t>() + col_offset / 2;
            auto *col_base =
                fp4_col_full.data_ptr<uint8_t>() + col_offset * (M / 2);
            alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
            create_tma_2d(
                tmap_in, inputs[g].data_ptr(), M, N_g,
                BUFF_DIM_Y, BUFF_DIM_X, inputs[g].stride(0), 16);
            create_tma_2d(
                tmap_out, row_base, M, N_g,
                BUFF_DIM_Y, BUFF_DIM_X, N_total, 4);
            create_tma_2d(
                tmap_out_t, col_base, N_g, M,
                BUFF_DIM_X, BUFF_DIM_Y, M, 4);

            launch_v2_kernel<true, false, true, true>(
                tmap_in, tmap_out, tmap_out_t,
                reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr()),
                reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr()),
                amax_tensor.data_ptr<float>() + g,
                amax_tensor.data_ptr<float>() + g,
                M, N_g, scale_stride, scale_stride_t, stream,
                nullptr, rng_state_ptr);

            if (g + 1 < G) {
                tk_advance_rng_state_kernel<<<1, 1, 0, stream>>>(
                    reinterpret_cast<unsigned long long*>(
                        rng_state.data_ptr<int64_t>()));
            }
            col_offset += N_g;
        }

        auto err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "tk_group_quantize_dim1_split3_for_gemm two-pass failed: ",
            cudaGetErrorString(err));

        std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
        std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
        std::vector<torch::Tensor> sc_row_u8_list(G), sc_col_u8_list(G);
        col_offset = 0;
        for (int g = 0; g < G; ++g) {
            const int64_t N_g = splits[g];
            fp4_row_list[g] =
                fp4_row_full.narrow(1, col_offset / 2, N_g / 2);
            sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
            sc_row_u8_list[g] = sc_row_allocs[g];
            fp4_col_list[g] =
                fp4_col_full.narrow(0, col_offset, N_g)
                    .view(torch::kFloat4_e2m1fn_x2);
            sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
            sc_col_u8_list[g] = sc_col_allocs[g];
            col_offset += N_g;
        }

        auto sc_row_cat =
            torch::cat(sc_row_u8_list, 1).view(torch::kFloat8_e4m3fn);
        auto sc_col_cat =
            torch::cat(sc_col_u8_list, 0).view(torch::kFloat8_e4m3fn);
        auto fp4_row_view = fp4_row_full.view(torch::kFloat4_e2m1fn_x2);
        auto fp4_col_view = fp4_col_full.view(torch::kFloat4_e2m1fn_x2);
        auto tma_dev_buf = torch::empty({0}, opts_u8);
        auto tma_host_buf = torch::empty(
            {0}, torch::dtype(torch::kUInt8).device(torch::kCPU));
        auto psync_tensor = torch::empty({0}, opts_u32);

        return std::make_tuple(
            fp4_row_list, sc_row_list, sg_per_group,
            fp4_col_list, sc_col_list,
            fp4_row_view, sc_row_cat,
            fp4_col_view, sc_col_cat,
            tma_dev_buf, tma_host_buf,
            rng_state, amax_tensor, sync_tensor, psync_tensor);
    }

    using namespace tk_v3;
    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;
    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;

    FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + G;
    args.num_groups   = G;
    args.sg_output    = sg_per_group.data_ptr<float>();
    args.swizzle_scales = true;
    args.rng_state = rng_state_ptr;
    args.split_range[0] = 0;

    std::array<int64_t, G> splits{n0, n1, n2};
    std::vector<torch::Tensor> sc_row_allocs(G), fp4_col_allocs(G), sc_col_allocs(G);
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = splits[g];
        args.split_range[g + 1] = args.split_range[g] + (int)N_g;
        args.blocks_per_group[g] = (int)(N_g / V3Config::CHUNK_DIM_X) * blocks_Y;

        const int64_t ntk_r_g = N_g / 64;
        sc_row_allocs[g] = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
        args.row_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());

        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        args.col_data_ptrs[g] = reinterpret_cast<fp4e2m1x2*>(fp4_col_allocs[g].data_ptr());

        const int64_t ntm_c_g = N_g / 128;
        const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
        sc_col_allocs[g] = torch::zeros({ntm_c_g, ntk_c, 512}, opts_u8);
        args.col_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)col_sc_stride;
    }

    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int fused_max_bps = any_data_sr
        ? ci.grp_d1_sr_max_bps
        : ci.grp_d1_max_bps;
    const int max_concurrent = fused_max_bps * persistent_launch_sms(ci);
    const bool use_fused = (total_grid <= max_concurrent && fused_max_bps > 0);

    auto tma_host_buf = torch::empty(
        {(int64_t)((3 + 2 * G) * (int64_t)sizeof(CUtensorMap))},
        torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    CUtensorMap* host_tma_maps = reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
    const auto input_l2 = use_fused
        ? CU_TENSOR_MAP_L2_PROMOTION_NONE
        : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    for (int g = 0; g < G; ++g) {
        create_tma_2d(host_tma_maps[g], inputs[g].data_ptr(), M, splits[g],
                      V3_BUFF_DIM_Y, V3_BUFF_DIM_X, inputs[g].stride(0), 16, input_l2);
    }
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = splits[g];
        const int64_t ntk_r_g = N_g / 64;
        const int64_t sc_row_x_bf16 = ntk_r_g * 256;
        create_tma_2d(host_tma_maps[3 + g], sc_row_allocs[g].data_ptr(),
                      ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        const int64_t ntm_c_g = N_g / 128;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(host_tma_maps[3 + G + g], sc_col_allocs[g].data_ptr(),
                      ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
    auto tma_dev_buf = torch::empty({(int64_t)((3 + 2 * G) * sizeof(CUtensorMap))}, opts_u8);
    cudaMemcpyAsync(tma_dev_buf.data_ptr(), host_tma_maps,
                    (3 + 2 * G) * sizeof(CUtensorMap),
                    cudaMemcpyHostToDevice, stream);
    const bool host_descriptor_recorded =
        at::cuda::CachingHostAllocator_recordEvent(
            tma_host_buf.data_ptr(),
            tma_host_buf.storage().data_ptr().get_context(),
            at::cuda::getCurrentCUDAStream(input0.get_device()));
    TORCH_CHECK(
        host_descriptor_recorded,
        "split3 TMA descriptor buffer was not allocated by the CUDA host allocator");
    const CUtensorMap* maps_dev = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
    args.input_tma_maps = maps_dev;
    args.scale_tma_maps = maps_dev + 3;
    args.use_tma_scales = true;

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;
    alignas(64) CUtensorMap tmap_in_dummy{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in_dummy, input0.data_ptr(), M, n0, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, input0.stride(0), 16, input_l2);
    create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
    create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    auto psync_tensor = torch::empty({0}, opts_u32);
    if (use_fused) {
        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;
        if (sr_axes.row && sr_axes.col) {
            fused_group_quantize_kernel_dim1<true, true><<<grid, V3_THREADS, dshmem, stream>>>(
                tmap_in_dummy, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        } else if (sr_axes.row) {
            fused_group_quantize_kernel_dim1<true, true, true, false><<<grid, V3_THREADS, dshmem, stream>>>(
                tmap_in_dummy, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        } else if (sr_axes.col) {
            fused_group_quantize_kernel_dim1<true, false, true, true><<<grid, V3_THREADS, dshmem, stream>>>(
                tmap_in_dummy, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        } else {
            fused_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
                tmap_in_dummy, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        }
    } else {
        psync_tensor = torch::empty({4}, opts_u32);
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());
        cudaMemsetAsync(psync, 0, 4 * sizeof(unsigned int), stream);

        const int persistent_max_bps = any_data_sr
            ? ci.pg_d1_sr_max_bps
            : ci.pg_d1_max_bps;
        int num_persistent = persistent_max_bps * persistent_launch_sms(ci);
        TORCH_CHECK(num_persistent > 0,
                    "split3 persistent grouped quantizer has zero resident blocks");
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = G;
        pargs.sg_output    = sg_per_group.data_ptr<float>();
        pargs.swizzle_scales = true;
        pargs.input_tma_maps = maps_dev;
        pargs.scale_tma_maps = maps_dev + 3;
        pargs.rng_state = rng_state_ptr;

        for (int g = 0; g <= G; ++g) pargs.split_range[g] = args.split_range[g];
        for (int g = 0; g < G; ++g) {
            pargs.blocks_per_group[g] = args.blocks_per_group[g];
            pargs.row_scale_ptrs[g]   = args.row_scale_ptrs[g];
            pargs.col_data_ptrs[g]    = args.col_data_ptrs[g];
            pargs.col_scale_ptrs[g]   = args.col_scale_ptrs[g];
            pargs.col_scale_stride[g] = args.col_scale_stride[g];
        }

        const int dshmem = ci.v3_dshmem_t;
        cudaError_t launch_err;
        if (barrier_free_group_quant_enabled()) {
            if (sr_axes.row && sr_axes.col) {
                launch_err = launch_persistent_group_quantize_two_pass<true, true, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            } else if (sr_axes.row) {
                launch_err = launch_persistent_group_quantize_two_pass<true, true, false>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            } else if (sr_axes.col) {
                launch_err = launch_persistent_group_quantize_two_pass<true, false, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            } else {
                launch_err = launch_persistent_group_quantize_two_pass<true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            }
        } else {
            if (sr_axes.row && sr_axes.col) {
                launch_err = launch_persistent_group_quantize_stream_safe<true, true, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            } else if (sr_axes.row) {
                launch_err = launch_persistent_group_quantize_stream_safe<true, true, false>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            } else if (sr_axes.col) {
                launch_err = launch_persistent_group_quantize_stream_safe<true, false, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            } else {
                launch_err = launch_persistent_group_quantize_stream_safe<true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, dshmem, stream);
            }
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "persistent split3 grouped quantizer launch failed: ",
                    cudaGetErrorString(launch_err));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_split3_for_gemm failed: ", cudaGetErrorString(err));

    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    std::vector<torch::Tensor> sc_row_u8_list(G), sc_col_u8_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = splits[g];
        fp4_row_list[g] = fp4_row_full.narrow(1, col_offset / 2, N_g / 2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_row_u8_list[g] = sc_row_allocs[g];
        fp4_col_list[g] = fp4_col_full.narrow(0, col_offset, N_g).view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_col_u8_list[g] = sc_col_allocs[g];
        col_offset += N_g;
    }

    auto sc_row_cat = torch::cat(sc_row_u8_list, 1).view(torch::kFloat8_e4m3fn);
    auto sc_col_cat = torch::cat(sc_col_u8_list, 0).view(torch::kFloat8_e4m3fn);
    auto fp4_row_view = fp4_row_full.view(torch::kFloat4_e2m1fn_x2);
    auto fp4_col_view = fp4_col_full.view(torch::kFloat4_e2m1fn_x2);

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list,
                           fp4_row_view, sc_row_cat,
                           fp4_col_view, sc_col_cat,
                           tma_dev_buf, tma_host_buf,
                           rng_state, amax_tensor, sync_tensor, psync_tensor);
}

struct Split3FrozenTensorMeta {
    std::string name;
    uintptr_t data_ptr = 0;
    c10::DeviceType device_type = c10::DeviceType::CPU;
    c10::DeviceIndex device_index = -1;
    c10::ScalarType scalar_type = c10::ScalarType::Undefined;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    bool contiguous = false;
    bool pinned = false;

    static Split3FrozenTensorMeta freeze(
        const torch::Tensor& tensor,
        std::string tensor_name
    ) {
        Split3FrozenTensorMeta meta;
        meta.name = std::move(tensor_name);
        meta.data_ptr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
        meta.device_type = tensor.device().type();
        meta.device_index = tensor.device().index();
        meta.scalar_type = tensor.scalar_type();
        meta.sizes = tensor.sizes().vec();
        meta.strides = tensor.strides().vec();
        meta.contiguous = tensor.is_contiguous();
        meta.pinned = tensor.device().is_cpu() && tensor.is_pinned();
        return meta;
    }

    void validate(const torch::Tensor& tensor) const {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) == data_ptr,
                    name, " pointer changed after split3 capture-state allocation");
        TORCH_CHECK(tensor.device().type() == device_type &&
                    tensor.device().index() == device_index,
                    name, " device changed after split3 capture-state allocation");
        TORCH_CHECK(tensor.scalar_type() == scalar_type,
                    name, " dtype changed after split3 capture-state allocation");
        TORCH_CHECK(tensor.sizes().vec() == sizes,
                    name, " shape changed after split3 capture-state allocation");
        TORCH_CHECK(tensor.strides().vec() == strides,
                    name, " stride changed after split3 capture-state allocation");
        TORCH_CHECK(tensor.is_contiguous() == contiguous,
                    name, " contiguity changed after split3 capture-state allocation");
        if (device_type == c10::DeviceType::CPU) {
            TORCH_CHECK(tensor.is_pinned() == pinned,
                        name, " pinned-storage contract changed after allocation");
        }
    }

    std::tuple<std::string, uint64_t, int64_t, int64_t, int64_t,
               std::vector<int64_t>, std::vector<int64_t>, bool, bool>
    manifest_row() const {
        return std::make_tuple(
            name, static_cast<uint64_t>(data_ptr),
            static_cast<int64_t>(device_type), static_cast<int64_t>(device_index),
            static_cast<int64_t>(scalar_type),
            sizes, strides, contiguous, pinned);
    }
};

class Split3CaptureState final {
public:
    std::array<torch::Tensor, 3> inputs;
    torch::Tensor fp4_row_full;
    torch::Tensor fp4_col_full;
    torch::Tensor sg_per_group;
    torch::Tensor amax_tensor;
    torch::Tensor sync_tensor;
    torch::Tensor psync_tensor;
    torch::Tensor rng_state;
    torch::Tensor tma_dev_buf;
    std::array<torch::Tensor, 3> sc_row_allocs;
    std::array<torch::Tensor, 3> fp4_col_aliases;
    std::array<torch::Tensor, 3> sc_col_allocs;
    torch::Tensor sc_row_cat;
    torch::Tensor sc_col_cat;
    torch::Tensor tma_host_buf;
    std::array<Split3FrozenTensorMeta, 3> frozen_inputs;
    std::vector<Split3FrozenTensorMeta> frozen_state;
    uintptr_t caller_stream = 0;
    bool data_stochastic_rounding = false;
    bool row_data_stochastic_rounding = false;
    bool col_data_stochastic_rounding = false;
    uint64_t rng_seed = 0;
    uint64_t rng_subsequence = 0;

    void validate(
        const std::array<torch::Tensor, 3>& launch_inputs,
        cudaStream_t stream
    ) const {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(stream) == caller_stream,
                    "split3 capture launch stream differs from allocation stream");
        for (int g = 0; g < 3; ++g) {
            frozen_inputs[g].validate(launch_inputs[g]);
            // Also validate the retained tensor object that owns each frozen
            // descriptor pointer; Python cannot replace these opaque fields.
            frozen_inputs[g].validate(inputs[g]);
        }
        TORCH_INTERNAL_ASSERT(frozen_state.size() == 20);
        size_t index = 0;
        frozen_state[index++].validate(fp4_row_full);
        frozen_state[index++].validate(fp4_col_full);
        frozen_state[index++].validate(sg_per_group);
        frozen_state[index++].validate(amax_tensor);
        frozen_state[index++].validate(sync_tensor);
        frozen_state[index++].validate(psync_tensor);
        frozen_state[index++].validate(rng_state);
        frozen_state[index++].validate(tma_dev_buf);
        for (const auto& tensor : sc_row_allocs) frozen_state[index++].validate(tensor);
        for (const auto& tensor : fp4_col_aliases) frozen_state[index++].validate(tensor);
        for (const auto& tensor : sc_col_allocs) frozen_state[index++].validate(tensor);
        frozen_state[index++].validate(sc_row_cat);
        frozen_state[index++].validate(sc_col_cat);
        frozen_state[index++].validate(tma_host_buf);
        TORCH_INTERNAL_ASSERT(index == frozen_state.size());
    }

    std::vector<std::tuple<std::string, uint64_t, int64_t, int64_t, int64_t,
                           std::vector<int64_t>, std::vector<int64_t>, bool, bool>>
    tensor_manifest() const {
        std::vector<std::tuple<std::string, uint64_t, int64_t, int64_t, int64_t,
                               std::vector<int64_t>, std::vector<int64_t>, bool, bool>>
            result;
        result.reserve(frozen_inputs.size() + frozen_state.size());
        for (const auto& meta : frozen_inputs) result.push_back(meta.manifest_row());
        for (const auto& meta : frozen_state) result.push_back(meta.manifest_row());
        return result;
    }

    uint64_t stream_manifest() const {
        return static_cast<uint64_t>(caller_stream);
    }
};

static inline void split3_require_exact_tensor(
    const torch::Tensor& tensor,
    const torch::Device& device,
    c10::ScalarType dtype,
    std::initializer_list<int64_t> sizes,
    const char* name
) {
    v5_require_tensor_device(tensor, device, name);
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == c10::IntArrayRef(sizes.begin(), sizes.size()),
                name, " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static inline void split3_require_exact_cpu_tensor(
    const torch::Tensor& tensor,
    c10::ScalarType dtype,
    std::initializer_list<int64_t> sizes,
    bool pinned,
    const char* name
) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must use CPU storage");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.sizes() == c10::IntArrayRef(sizes.begin(), sizes.size()),
                name, " has the wrong shape");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.is_pinned() == pinned, name, " has the wrong pinned contract");
}

struct Split3MemoryRows {
    uintptr_t base;
    uintptr_t row_stride_bytes;
    uintptr_t row_bytes;
    int64_t rows;
};

static Split3MemoryRows split3_memory_rows(const torch::Tensor& tensor) {
    const uintptr_t element_bytes = tensor.element_size();
    if (tensor.is_contiguous()) {
        return {reinterpret_cast<uintptr_t>(tensor.data_ptr()), 0,
                static_cast<uintptr_t>(tensor.numel()) * element_bytes, 1};
    }
    TORCH_INTERNAL_ASSERT(tensor.dim() == 2 && tensor.stride(1) == 1 &&
                          tensor.stride(0) >= tensor.size(1));
    return {reinterpret_cast<uintptr_t>(tensor.data_ptr()),
            static_cast<uintptr_t>(tensor.stride(0)) * element_bytes,
            static_cast<uintptr_t>(tensor.size(1)) * element_bytes,
            tensor.size(0)};
}

static bool split3_tensors_overlap(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs
) {
    const auto a = split3_memory_rows(lhs);
    const auto b = split3_memory_rows(rhs);
    int64_t i = 0;
    int64_t j = 0;
    while (i < a.rows && j < b.rows) {
        const uintptr_t a_begin = a.base + static_cast<uintptr_t>(i) * a.row_stride_bytes;
        const uintptr_t a_end = a_begin + a.row_bytes;
        const uintptr_t b_begin = b.base + static_cast<uintptr_t>(j) * b.row_stride_bytes;
        const uintptr_t b_end = b_begin + b.row_bytes;
        if (a_end <= b_begin) {
            ++i;
        } else if (b_end <= a_begin) {
            ++j;
        } else {
            return true;
        }
    }
    return false;
}

static void split3_validate_alias_and_overlap(
    const Split3CaptureState& state,
    const std::array<torch::Tensor, 3>& inputs
) {
    const std::array<const torch::Tensor*, 19> independent{
        &inputs[0], &inputs[1], &inputs[2],
        &state.fp4_row_full, &state.fp4_col_full, &state.sg_per_group,
        &state.amax_tensor, &state.sync_tensor, &state.psync_tensor,
        &state.rng_state,
        &state.tma_dev_buf,
        &state.sc_row_allocs[0], &state.sc_row_allocs[1], &state.sc_row_allocs[2],
        &state.sc_col_allocs[0], &state.sc_col_allocs[1], &state.sc_col_allocs[2],
        &state.sc_row_cat, &state.sc_col_cat};
    for (size_t lhs = 0; lhs < independent.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < independent.size(); ++rhs) {
            TORCH_CHECK(!split3_tensors_overlap(*independent[lhs], *independent[rhs]),
                        "split3 capture state has forbidden input/output/source overlap at ",
                        lhs, " and ", rhs);
        }
    }

    int64_t col_offset = 0;
    const int64_t col_stride = state.fp4_col_full.size(1);
    for (int g = 0; g < 3; ++g) {
        const auto& alias = state.fp4_col_aliases[g];
        const uintptr_t expected =
            reinterpret_cast<uintptr_t>(state.fp4_col_full.data_ptr()) +
            static_cast<uintptr_t>(col_offset * col_stride);
        TORCH_CHECK(reinterpret_cast<uintptr_t>(alias.data_ptr()) == expected,
                    "split3 fp4_col alias ", g,
                    " is not the exact contiguous slice of fp4_col_full");
        for (size_t other = 0; other < independent.size(); ++other) {
            if (other == 4) continue;  // Exact containment in fp4_col_full is required.
            TORCH_CHECK(!split3_tensors_overlap(alias, *independent[other]),
                        "split3 fp4_col alias overlaps a non-owner tensor at ", other);
        }
        for (int prior = 0; prior < g; ++prior) {
            TORCH_CHECK(!split3_tensors_overlap(alias, state.fp4_col_aliases[prior]),
                        "split3 fp4_col aliases overlap");
        }
        col_offset += alias.size(0);
    }
    TORCH_CHECK(col_offset == state.fp4_col_full.size(0),
                "split3 fp4_col aliases do not exactly partition fp4_col_full");
}

// Build a capture state from an ordinary eager split3 result.  This function
// is intentionally allocation-only and must run before capture.  Both input
// and scale TMA descriptors are encoded for the exact retained pointers and
// copied to device here; the launch entrypoint neither rewrites pinned host
// memory nor adds a descriptor H2D node to the graph.
std::shared_ptr<Split3CaptureState> tk_group_quantize_dim1_split3_capture_alloc(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor fp4_row_full,
    torch::Tensor fp4_col_full,
    torch::Tensor sg_per_group,
    std::vector<torch::Tensor> sc_row_allocs,
    std::vector<torch::Tensor> fp4_col_allocs,
    std::vector<torch::Tensor> sc_col_allocs,
    torch::Tensor sc_row_cat,
    torch::Tensor sc_col_cat,
    torch::Tensor tma_dev_buf,
    torch::Tensor tma_host_buf,
    bool data_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    std::string data_sr_axes
) {
    std::array<torch::Tensor, 3> inputs{input0, input1, input2};
    for (const auto& input : inputs) {
        TORCH_CHECK(input.is_cuda(), "split3 capture alloc inputs must be CUDA");
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                    "split3 capture alloc inputs must be 2D BF16 tensors");
        TORCH_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1),
                    "split3 capture alloc inputs must be non-overlapping and inner-contiguous");
    }
    TORCH_CHECK(input0.device() == input1.device() && input0.device() == input2.device(),
                "split3 capture alloc inputs must share a CUDA device");
    const auto device = input0.device();
    const c10::cuda::CUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(input0.get_device()).stream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    v5_check_cuda(cudaStreamIsCapturing(stream, &capture_status),
                  "split3 capture alloc stream query failed");
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "tk_group_quantize_dim1_split3_capture_alloc must run before capture");
    const auto sr_axes = resolve_data_sr_axes(
        data_stochastic_rounding, std::move(data_sr_axes), "split3 capture");

    const int64_t M = input0.size(0);
    TORCH_CHECK(input1.size(0) == M && input2.size(0) == M,
                "split3 capture alloc inputs must share rows");
    const int64_t n0 = input0.size(1), n1 = input1.size(1), n2 = input2.size(1);
    const int64_t N_total = n0 + n1 + n2;
    TORCH_CHECK(M % 128 == 0 && N_total % 128 == 0 &&
                n0 % 128 == 0 && n1 % 128 == 0 && n2 % 128 == 0,
                "split3 capture alloc dimensions must be multiples of 128");
    constexpr int G = 3;
    TORCH_CHECK(sc_row_allocs.size() == G && fp4_col_allocs.size() == G &&
                sc_col_allocs.size() == G,
                "split3 capture alloc expects three output groups");
    split3_require_exact_tensor(
        fp4_row_full, device, torch::kUInt8, {M, N_total / 2}, "fp4_row_full");
    split3_require_exact_tensor(
        fp4_col_full, device, torch::kUInt8, {N_total, M / 2}, "fp4_col_full");
    split3_require_exact_tensor(
        sg_per_group, device, torch::kFloat32, {G}, "sg_per_group");
    split3_require_exact_tensor(
        tma_dev_buf, device, torch::kUInt8,
        {(int64_t)((3 + 2 * G) * sizeof(CUtensorMap))}, "tma_dev_buf");
    split3_require_exact_cpu_tensor(
        tma_host_buf, torch::kUInt8,
        {(int64_t)((3 + 2 * G) * sizeof(CUtensorMap))}, true, "tma_host_buf");

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;
    std::array<int64_t, G> splits{n0, n1, n2};
    int64_t row_cat_width = 0;
    int64_t col_cat_rows = 0;
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t ntk_r_g = splits[g] / 64;
        const int64_t ntm_c_g = splits[g] / 128;
        const std::string row_name = "sc_row_allocs[" + std::to_string(g) + "]";
        const std::string col_alias_name = "fp4_col_aliases[" + std::to_string(g) + "]";
        const std::string col_name = "sc_col_allocs[" + std::to_string(g) + "]";
        split3_require_exact_tensor(
            sc_row_allocs[g], device, torch::kUInt8,
            {ntm_r, ntk_r_g, 512}, row_name.c_str());
        split3_require_exact_tensor(
            fp4_col_allocs[g], device, torch::kUInt8,
            {splits[g], M / 2}, col_alias_name.c_str());
        split3_require_exact_tensor(
            sc_col_allocs[g], device, torch::kUInt8,
            {ntm_c_g, ntk_c, 512}, col_name.c_str());
        const uintptr_t expected_col_ptr =
            reinterpret_cast<uintptr_t>(fp4_col_full.data_ptr()) +
            static_cast<uintptr_t>(col_offset * (M / 2));
        TORCH_CHECK(reinterpret_cast<uintptr_t>(fp4_col_allocs[g].data_ptr()) ==
                        expected_col_ptr,
                    "split3 capture alloc fp4_col alias ", g,
                    " must be the exact contiguous fp4_col_full slice");
        row_cat_width += ntk_r_g;
        col_cat_rows += ntm_c_g;
        col_offset += splits[g];
    }
    split3_require_exact_tensor(
        sc_row_cat, device, torch::kUInt8,
        {ntm_r, row_cat_width, 512}, "sc_row_cat");
    split3_require_exact_tensor(
        sc_col_cat, device, torch::kUInt8,
        {col_cat_rows, ntk_c, 512}, "sc_col_cat");

    // The eager split3 call may still be reading its descriptor buffer.  This
    // one-time graph warmup synchronization makes re-encoding that retained
    // bundle unambiguous without affecting ordinary graph-off production.
    v5_check_cuda(cudaStreamSynchronize(stream),
                  "split3 capture alloc warmup synchronization failed");

    using namespace tk_v3;
    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int fused_max_bps = data_stochastic_rounding
        ? ci.grp_d1_sr_max_bps
        : ci.grp_d1_max_bps;
    const int max_concurrent = fused_max_bps * persistent_launch_sms(ci);
    const bool use_fused = total_grid <= max_concurrent && fused_max_bps > 0;
    const auto input_l2 = use_fused
        ? CU_TENSOR_MAP_L2_PROMOTION_NONE
        : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;

    CUtensorMap* host_tma_maps =
        reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
    for (int g = 0; g < G; ++g) {
        create_tma_2d(host_tma_maps[g], inputs[g].data_ptr(), M, splits[g],
                      V3_BUFF_DIM_Y, V3_BUFF_DIM_X, inputs[g].stride(0), 16,
                      input_l2);
        const int64_t ntk_r_g = splits[g] / 64;
        const int64_t sc_row_x_bf16 = ntk_r_g * 256;
        create_tma_2d(host_tma_maps[3 + g], sc_row_allocs[g].data_ptr(),
                      ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        const int64_t ntm_c_g = splits[g] / 128;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(host_tma_maps[3 + G + g], sc_col_allocs[g].data_ptr(),
                      ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
    v5_check_cuda(cudaMemcpy(
        tma_dev_buf.data_ptr(), host_tma_maps,
        (3 + 2 * G) * sizeof(CUtensorMap), cudaMemcpyHostToDevice),
        "split3 capture alloc descriptor copy failed");

    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto amax_tensor = torch::empty({G}, opts_f32);
    auto sync_tensor = torch::empty({2 * G}, opts_u32);
    auto psync_tensor = torch::empty({4}, opts_u32);
    auto rng_state = torch::empty(
        {2}, torch::dtype(torch::kInt64).device(device));
    split3_require_exact_tensor(
        amax_tensor, device, torch::kFloat32, {G}, "amax_tensor");
    split3_require_exact_tensor(
        sync_tensor, device, torch::kInt32, {2 * G}, "sync_tensor");
    split3_require_exact_tensor(
        psync_tensor, device, torch::kInt32, {4}, "psync_tensor");
    split3_require_exact_tensor(
        rng_state, device, torch::kInt64, {2}, "rng_state");

    auto state = std::make_shared<Split3CaptureState>();
    state->inputs = inputs;
    state->fp4_row_full = fp4_row_full;
    state->fp4_col_full = fp4_col_full;
    state->sg_per_group = sg_per_group;
    state->amax_tensor = amax_tensor;
    state->sync_tensor = sync_tensor;
    state->psync_tensor = psync_tensor;
    state->rng_state = rng_state;
    state->tma_dev_buf = tma_dev_buf;
    for (int g = 0; g < G; ++g) {
        state->sc_row_allocs[g] = sc_row_allocs[g];
        state->fp4_col_aliases[g] = fp4_col_allocs[g];
        state->sc_col_allocs[g] = sc_col_allocs[g];
        state->frozen_inputs[g] = Split3FrozenTensorMeta::freeze(
            inputs[g], "input[" + std::to_string(g) + "]");
    }
    state->sc_row_cat = sc_row_cat;
    state->sc_col_cat = sc_col_cat;
    state->tma_host_buf = tma_host_buf;
    state->caller_stream = reinterpret_cast<uintptr_t>(stream);
    state->row_data_stochastic_rounding = sr_axes.row;
    state->col_data_stochastic_rounding = sr_axes.col;
    state->data_stochastic_rounding = sr_axes.row || sr_axes.col;
    state->rng_seed = rng_seed;
    state->rng_subsequence = rng_subsequence;

    auto freeze_state = [&](const torch::Tensor& tensor, std::string name) {
        state->frozen_state.push_back(
            Split3FrozenTensorMeta::freeze(tensor, std::move(name)));
    };
    state->frozen_state.reserve(20);
    freeze_state(state->fp4_row_full, "fp4_row_full");
    freeze_state(state->fp4_col_full, "fp4_col_full");
    freeze_state(state->sg_per_group, "sg_per_group");
    freeze_state(state->amax_tensor, "amax_tensor");
    freeze_state(state->sync_tensor, "sync_tensor");
    freeze_state(state->psync_tensor, "psync_tensor");
    freeze_state(state->rng_state, "rng_state");
    freeze_state(state->tma_dev_buf, "tma_dev_buf");
    for (int g = 0; g < G; ++g) {
        freeze_state(state->sc_row_allocs[g],
                     "sc_row_allocs[" + std::to_string(g) + "]");
    }
    for (int g = 0; g < G; ++g) {
        freeze_state(state->fp4_col_aliases[g],
                     "fp4_col_aliases[" + std::to_string(g) + "]");
    }
    for (int g = 0; g < G; ++g) {
        freeze_state(state->sc_col_allocs[g],
                     "sc_col_allocs[" + std::to_string(g) + "]");
    }
    freeze_state(state->sc_row_cat, "sc_row_cat");
    freeze_state(state->sc_col_cat, "sc_col_cat");
    freeze_state(state->tma_host_buf, "tma_host_buf");
    split3_validate_alias_and_overlap(*state, inputs);
    return state;
}

static void split3_copy_scale_cats(
    const std::array<torch::Tensor, 3>& sc_row_allocs,
    const std::array<torch::Tensor, 3>& sc_col_allocs,
    torch::Tensor sc_row_cat,
    torch::Tensor sc_col_cat,
    int64_t M,
    const std::array<int64_t, 3>& splits,
    cudaStream_t stream
) {
    const int64_t ntm_r = M / 128;
    const int64_t total_row_width =
        splits[0] / 64 + splits[1] / 64 + splits[2] / 64;
    size_t row_offset_bytes = 0;
    size_t col_offset_bytes = 0;
    for (int g = 0; g < 3; ++g) {
        const size_t row_width_bytes = static_cast<size_t>(splits[g] / 64) * 512;
        v5_check_cuda(cudaMemcpy2DAsync(
            static_cast<char*>(sc_row_cat.data_ptr()) + row_offset_bytes,
            static_cast<size_t>(total_row_width) * 512,
            sc_row_allocs[g].data_ptr(), row_width_bytes,
            row_width_bytes, static_cast<size_t>(ntm_r),
            cudaMemcpyDeviceToDevice, stream),
            "split3 row-scale cat copy failed");
        row_offset_bytes += row_width_bytes;

        const size_t col_bytes = sc_col_allocs[g].numel();
        v5_check_cuda(cudaMemcpyAsync(
            static_cast<char*>(sc_col_cat.data_ptr()) + col_offset_bytes,
            sc_col_allocs[g].data_ptr(), col_bytes,
            cudaMemcpyDeviceToDevice, stream),
            "split3 col-scale cat copy failed");
        col_offset_bytes += col_bytes;
    }
}

// Allocation-free capture launch.  Exact input pointers, shapes, and strides
// must match the descriptors frozen by the alloc call.  Output cat tensors are
// also preallocated and refreshed with graph-capturable D2D copies.
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_group_quantize_dim1_split3_launch(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    const std::shared_ptr<Split3CaptureState>& state
) {
    TORCH_CHECK(state, "split3 capture launch requires an opaque native state");
    std::array<torch::Tensor, 3> inputs{input0, input1, input2};
    for (const auto& input : inputs) {
        TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 &&
                    input.dim() == 2 && input.stride(1) == 1 &&
                    input.stride(0) >= input.size(1),
                    "split3 capture launch requires inner-contiguous CUDA BF16 inputs");
    }
    TORCH_CHECK(input0.device() == input1.device() && input0.device() == input2.device(),
                "split3 capture launch inputs must share a CUDA device");
    const auto device = input0.device();
    const c10::cuda::CUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(input0.get_device()).stream();
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    v5_check_cuda(cudaStreamIsCapturing(stream, &capture_status),
                  "split3 capture launch stream query failed");
    TORCH_CHECK(capture_status != cudaStreamCaptureStatusNone,
                "tk_group_quantize_dim1_split3_launch is capture-only");
    state->validate(inputs, stream);
    split3_validate_alias_and_overlap(*state, inputs);

    const auto& fp4_row_full = state->fp4_row_full;
    const auto& fp4_col_full = state->fp4_col_full;
    const auto& sg_per_group = state->sg_per_group;
    const auto& amax_tensor = state->amax_tensor;
    const auto& sync_tensor = state->sync_tensor;
    const auto& psync_tensor = state->psync_tensor;
    const auto& rng_state = state->rng_state;
    const auto& tma_dev_buf = state->tma_dev_buf;
    const auto& sc_row_allocs = state->sc_row_allocs;
    const auto& sc_col_allocs = state->sc_col_allocs;
    const auto& sc_row_cat = state->sc_row_cat;
    const auto& sc_col_cat = state->sc_col_cat;
    const auto& tma_host_buf = state->tma_host_buf;

    const int64_t M = input0.size(0);
    const int64_t n0 = input0.size(1), n1 = input1.size(1), n2 = input2.size(1);
    const int64_t N_total = n0 + n1 + n2;
    constexpr int G = 3;
    std::array<int64_t, G> splits{n0, n1, n2};

    V5QuantSequenceGuard sequence_guard(stream);
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);
    cudaMemsetAsync(psync_tensor.data_ptr(), 0, 4 * sizeof(unsigned int), stream);
    if (state->data_stochastic_rounding) {
        tk_prepare_advancing_rng_state_kernel<<<1, 1, 0, stream>>>(
            reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
            static_cast<unsigned long long>(state->rng_seed),
            static_cast<unsigned long long>(state->rng_subsequence),
            1ull);
    }

    using namespace tk_v3;
    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;
    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;
    FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag = args.done_counter + G;
    args.num_groups = G;
    args.sg_output = sg_per_group.data_ptr<float>();
    args.swizzle_scales = true;
    args.rng_state = state->data_stochastic_rounding
        ? reinterpret_cast<const uint64_t*>(rng_state.data_ptr<int64_t>())
        : nullptr;
    args.split_range[0] = 0;
    for (int g = 0; g < G; ++g) {
        args.split_range[g + 1] = args.split_range[g] + (int)splits[g];
        args.blocks_per_group[g] =
            (int)(splits[g] / V3Config::CHUNK_DIM_X) * blocks_Y;
        args.row_scale_ptrs[g] =
            reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());
        args.col_data_ptrs[g] = nullptr;
        args.col_scale_ptrs[g] =
            reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)(((M / 16) + 3) / 4 * 4);
    }

    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int fused_max_bps = state->data_stochastic_rounding
        ? ci.grp_d1_sr_max_bps
        : ci.grp_d1_max_bps;
    const int max_concurrent = fused_max_bps * persistent_launch_sms(ci);
    const bool use_fused = total_grid <= max_concurrent && fused_max_bps > 0;
    const auto input_l2 = use_fused
        ? CU_TENSOR_MAP_L2_PROMOTION_NONE
        : CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    const CUtensorMap* maps_dev =
        reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
    args.input_tma_maps = maps_dev;
    args.scale_tma_maps = maps_dev + 3;
    args.use_tma_scales = true;

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;
    alignas(64) CUtensorMap tmap_in_dummy{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in_dummy, input0.data_ptr(), M, n0,
                  V3_BUFF_DIM_Y, V3_BUFF_DIM_X, input0.stride(0), 16,
                  input_l2);
    create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total,
                  V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
    create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M,
                  V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    if (use_fused) {
        const dim3 grid(blocks_X, blocks_Y);
        if (state->row_data_stochastic_rounding &&
            state->col_data_stochastic_rounding) {
            launch_manual_barrier_kernel_capture_safe(
                "fused grouped dim1 split3 SR capture quantizer",
                fused_group_quantize_kernel_dim1<true, true>,
                grid, dim3(V3_THREADS), ci.v3_dshmem_t, stream,
                tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        } else if (state->row_data_stochastic_rounding) {
            launch_manual_barrier_kernel_capture_safe(
                "fused grouped dim1 split3 row-SR capture quantizer",
                fused_group_quantize_kernel_dim1<true, true, true, false>,
                grid, dim3(V3_THREADS), ci.v3_dshmem_t, stream,
                tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        } else if (state->col_data_stochastic_rounding) {
            launch_manual_barrier_kernel_capture_safe(
                "fused grouped dim1 split3 col-SR capture quantizer",
                fused_group_quantize_kernel_dim1<true, false, true, true>,
                grid, dim3(V3_THREADS), ci.v3_dshmem_t, stream,
                tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        } else {
            launch_manual_barrier_kernel_capture_safe(
                "fused grouped dim1 split3 capture quantizer",
                fused_group_quantize_kernel_dim1<true>,
                grid, dim3(V3_THREADS), ci.v3_dshmem_t, stream,
                tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                args);
        }
    } else {
        auto* psync =
            reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());
        const int persistent_max_bps = state->data_stochastic_rounding
            ? ci.pg_d1_sr_max_bps
            : ci.pg_d1_max_bps;
        int num_persistent = persistent_max_bps * persistent_launch_sms(ci);
        TORCH_CHECK(num_persistent > 0,
                    "split3 capture persistent quantizer has zero resident blocks");
        if (num_persistent > total_grid) num_persistent = total_grid;
        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag = psync + 3;
        pargs.tiles_X = blocks_X;
        pargs.tiles_Y = blocks_Y;
        pargs.total_tiles = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups = G;
        pargs.sg_output = sg_per_group.data_ptr<float>();
        pargs.swizzle_scales = true;
        pargs.input_tma_maps = maps_dev;
        pargs.scale_tma_maps = maps_dev + 3;
        pargs.rng_state = args.rng_state;
        for (int g = 0; g <= G; ++g) pargs.split_range[g] = args.split_range[g];
        for (int g = 0; g < G; ++g) {
            pargs.blocks_per_group[g] = args.blocks_per_group[g];
            pargs.row_scale_ptrs[g] = args.row_scale_ptrs[g];
            pargs.col_data_ptrs[g] = args.col_data_ptrs[g];
            pargs.col_scale_ptrs[g] = args.col_scale_ptrs[g];
            pargs.col_scale_stride[g] = args.col_scale_stride[g];
        }
        cudaError_t launch_err;
        if (barrier_free_group_quant_enabled()) {
            if (state->row_data_stochastic_rounding &&
                state->col_data_stochastic_rounding) {
                launch_err = launch_persistent_group_quantize_two_pass<true, true, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            } else if (state->row_data_stochastic_rounding) {
                launch_err = launch_persistent_group_quantize_two_pass<true, true, false>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            } else if (state->col_data_stochastic_rounding) {
                launch_err = launch_persistent_group_quantize_two_pass<true, false, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            } else {
                launch_err = launch_persistent_group_quantize_two_pass<true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            }
        } else {
            if (state->row_data_stochastic_rounding &&
                state->col_data_stochastic_rounding) {
                launch_err = launch_persistent_group_quantize_stream_safe<true, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            } else if (state->row_data_stochastic_rounding) {
                launch_err = launch_persistent_group_quantize_stream_safe<true, true, false>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            } else if (state->col_data_stochastic_rounding) {
                launch_err = launch_persistent_group_quantize_stream_safe<true, false, true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            } else {
                launch_err = launch_persistent_group_quantize_stream_safe<true>(
                    tmap_in_dummy, tmap_out, tmap_out_t, nullptr,
                    M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                    pargs, num_persistent, ci.v3_dshmem_t, stream);
            }
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "persistent grouped dim1 split3 capture launch failed: ",
                    cudaGetErrorString(launch_err));
    }
    v5_check_cuda(cudaGetLastError(),
                  "tk_group_quantize_dim1_split3_launch failed");
    split3_copy_scale_cats(
        sc_row_allocs, sc_col_allocs, sc_row_cat, sc_col_cat,
        M, splits, stream);

    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        fp4_row_list[g] = fp4_row_full.narrow(
            1, col_offset / 2, splits[g] / 2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
        fp4_col_list[g] = fp4_col_full.narrow(
            0, col_offset, splits[g]).view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
        col_offset += splits[g];
    }
    return std::make_tuple(
        fp4_row_list, sc_row_list, sg_per_group,
        fp4_col_list, sc_col_list,
        fp4_row_full.view(torch::kFloat4_e2m1fn_x2),
        sc_row_cat.view(torch::kFloat8_e4m3fn),
        fp4_col_full.view(torch::kFloat4_e2m1fn_x2),
        sc_col_cat.view(torch::kFloat8_e4m3fn),
        tma_dev_buf, tma_host_buf);
}


// ═══════════════════════════════════════════════════════════════════
// 3e. Dim-1 split API for CUDA graph capture:
//     - dim1_alloc: pre-allocate all output tensors (call BEFORE graph capture)
//     - dim1_launch: kernel-only dispatch (safe INSIDE graph capture)
// ═══════════════════════════════════════════════════════════════════

// Returns: (fp4_row_full, fp4_col_full, sg_per_group,
//           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
//           sc_row_allocs, fp4_col_allocs, sc_col_allocs, tma_host_buf)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, torch::Tensor>
tk_group_quantize_dim1_alloc(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    const int64_t M = input.size(0), N_total = input.size(1);
    const int G = (int)col_split_sections.size();
    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    auto fp4_row_full = torch::empty({M, N_total / 2}, opts_u8);
    auto fp4_col_full = torch::empty({N_total, M / 2}, opts_u8);
    auto sg_per_group = torch::empty({G}, opts_f32);
    auto amax_tensor  = torch::empty({G}, opts_f32);
    auto sync_tensor  = torch::empty({2 * G}, opts_u32);
    auto psync_tensor = torch::empty({4}, opts_u32);
    auto tma_dev_buf  = torch::empty({(int64_t)(2 * G * (int64_t)sizeof(CUtensorMap))}, opts_u8);

    // Dedicated pinned host buffer for TMA descriptors — NOT shared with
    // other kernels, so it won't be overwritten between CUDA graph replays.
    auto tma_host_buf = torch::empty({(int64_t)(2 * G * (int64_t)sizeof(CUtensorMap))},
                                     torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));

    std::vector<torch::Tensor> sc_row_allocs(G), fp4_col_allocs(G), sc_col_allocs(G);
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        const int64_t ntk_r_g = N_g / 64;
        sc_row_allocs[g]  = torch::empty({ntm_r, ntk_r_g, 512}, opts_u8);
        fp4_col_allocs[g] = torch::empty({N_g, M / 2}, opts_u8);
        const int64_t ntm_c_g = N_g / 128;
        sc_col_allocs[g]  = torch::zeros({ntm_c_g, ntk_c, 512}, opts_u8);
    }

    return std::make_tuple(fp4_row_full, fp4_col_full, sg_per_group,
                           amax_tensor, sync_tensor, psync_tensor, tma_dev_buf,
                           sc_row_allocs, fp4_col_allocs, sc_col_allocs,
                           tma_host_buf);
}

// Kernel-only dispatch — NO allocations, safe inside CUDA graph capture.
// Takes pre-allocated buffers from dim1_alloc.
// Returns: (fp4_row_list, sc_row_list, sg_per_group,
//           fp4_col_list, sc_col_list,
//           fp4_row_view, sc_row_cat, fp4_col_view, sc_col_cat)
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_group_quantize_dim1_launch(
    torch::Tensor input,
    std::vector<int64_t> col_split_sections,
    // Pre-allocated buffers from dim1_alloc:
    torch::Tensor fp4_row_full,
    torch::Tensor fp4_col_full,
    torch::Tensor sg_per_group,
    torch::Tensor amax_tensor,
    torch::Tensor sync_tensor,
    torch::Tensor psync_tensor,
    torch::Tensor tma_host_buf,  // dedicated pinned host buffer for TMA descs
    torch::Tensor tma_dev_buf,
    std::vector<torch::Tensor> sc_row_allocs,
    std::vector<torch::Tensor> fp4_col_allocs,
    std::vector<torch::Tensor> sc_col_allocs,
    bool skip_cat = false
) {
    const int64_t M = input.size(0), N_total = input.size(1);
    const int G = (int)col_split_sections.size();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

    const int64_t ntm_r = M / 128;
    const int64_t ntk_c = M / 64;

    // Zero sync buffers (cudaMemsetAsync is graph-safe)
    cudaMemsetAsync(amax_tensor.data_ptr(), 0, G * sizeof(float), stream);
    cudaMemsetAsync(sync_tensor.data_ptr(), 0, 2 * G * sizeof(unsigned int), stream);
    cudaMemsetAsync(psync_tensor.data_ptr(), 0, 4 * sizeof(unsigned int), stream);
    // Zero sc_col_allocs (they use torch::zeros in alloc, but need re-zeroing for graph replay)
    for (int g = 0; g < G; ++g) {
        cudaMemsetAsync(sc_col_allocs[g].data_ptr(), 0,
                        sc_col_allocs[g].numel() * sc_col_allocs[g].element_size(), stream);
    }

    using namespace tk_v3;
    FusedGroupArgs args;
    memset(&args, 0, sizeof(args));
    args.global_amax  = amax_tensor.data_ptr<float>();
    args.done_counter = reinterpret_cast<unsigned int*>(sync_tensor.data_ptr<int32_t>());
    args.ready_flag   = args.done_counter + G;
    args.num_groups   = G;
    args.sg_output    = sg_per_group.data_ptr<float>();
    args.fwd_b_sg     = nullptr;
    args.dgrad_b_sg   = nullptr;
    args.swizzle_scales = true;
    args.split_range[0] = 0;

    const int blocks_Y = M / V3Config::CHUNK_DIM_Y;

    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        args.split_range[g + 1] = args.split_range[g] + (int)N_g;
        args.blocks_per_group[g] = (int)(N_g / V3Config::CHUNK_DIM_X) * blocks_Y;

        args.row_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_row_allocs[g].data_ptr());
        args.col_data_ptrs[g] = reinterpret_cast<fp4e2m1x2*>(fp4_col_allocs[g].data_ptr());
        const int64_t col_sc_stride = ((M / 16) + 3) / 4 * 4;
        args.col_scale_ptrs[g] = reinterpret_cast<nvfp4_scale_t*>(sc_col_allocs[g].data_ptr());
        args.col_scale_stride[g] = (int)col_sc_stride;
    }

    const int blocks_X = N_total / V3Config::CHUNK_DIM_X;
    const int total_grid = blocks_X * blocks_Y;
    const auto& ci = get_cached_info();
    const int max_concurrent = ci.grp_d1_max_bps * persistent_launch_sms(ci);
    const bool use_fused = (total_grid <= max_concurrent && ci.grp_d1_max_bps > 0);

    const int64_t global_scale_stride = ((N_total / 16) + 3) / 4 * 4;

    // Create TMA maps in DEDICATED pinned host buffer (not the shared one),
    // then copy to device. The dedicated buffer isn't overwritten by other
    // kernels between CUDA graph replays, so cudaMemcpyAsync reads correct data.
    CUtensorMap* pinned_maps = reinterpret_cast<CUtensorMap*>(tma_host_buf.data_ptr());
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        const int64_t ntk_r_g = N_g / 64;
        const int64_t sc_row_x_bf16 = ntk_r_g * 256;
        create_tma_2d(pinned_maps[g], sc_row_allocs[g].data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);

        const int64_t ntm_c_g = N_g / 128;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(pinned_maps[G + g], sc_col_allocs[g].data_ptr(), ntm_c_g, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }
    cudaMemcpyAsync(tma_dev_buf.data_ptr(), pinned_maps, 2 * G * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    if (use_fused) {
        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        args.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());
        args.use_tma_scales = true;

        const dim3 grid(blocks_X, blocks_Y);
        const int dshmem = ci.v3_dshmem_t;
        fused_group_quantize_kernel_dim1<true><<<grid, V3_THREADS, dshmem, stream>>>(
            tmap_in, tmap_out, tmap_out_t,
            nullptr,
            M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
            args);
    } else {
        // Persistent grouped path
        unsigned int *psync = reinterpret_cast<unsigned int*>(psync_tensor.data_ptr<int32_t>());

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_in, input.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out, fp4_row_full.data_ptr(), M, N_total, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, N_total, 4);
        create_tma_2d(tmap_out_t, fp4_col_full.data_ptr(), N_total, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        int num_persistent = ci.pg_d1_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_grid) num_persistent = total_grid;

        tk_v5::PersistentGroupArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = psync;
        pargs.work_counter_phase2 = psync + 1;
        pargs.global_amax  = amax_tensor.data_ptr<float>();
        pargs.done_counter = psync + 2;
        pargs.ready_flag   = psync + 3;
        pargs.tiles_X      = blocks_X;
        pargs.tiles_Y      = blocks_Y;
        pargs.total_tiles  = total_grid;
        pargs.num_persistent = num_persistent;
        pargs.num_groups   = G;
        pargs.sg_output    = sg_per_group.data_ptr<float>();
        pargs.fwd_b_sg     = nullptr;
        pargs.dgrad_b_sg   = nullptr;
        pargs.swizzle_scales = true;

        for (int g = 0; g <= G; ++g) pargs.split_range[g] = args.split_range[g];
        for (int g = 0; g < G; ++g) {
            pargs.blocks_per_group[g] = args.blocks_per_group[g];
            pargs.row_scale_ptrs[g]   = args.row_scale_ptrs[g];
            pargs.col_data_ptrs[g]    = args.col_data_ptrs[g];
            pargs.col_scale_ptrs[g]   = args.col_scale_ptrs[g];
            pargs.col_scale_stride[g] = args.col_scale_stride[g];
        }
        pargs.scale_tma_maps = reinterpret_cast<const CUtensorMap*>(tma_dev_buf.data_ptr());

        const int dshmem = ci.v3_dshmem_t;
        cudaError_t launch_err;
        if (barrier_free_group_quant_enabled()) {
            launch_err = launch_persistent_group_quantize_two_pass<true>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        } else {
            launch_err = launch_persistent_group_quantize_stream_safe<true>(
                tmap_in, tmap_out, tmap_out_t,
                nullptr,
                M, N_total, global_scale_stride, ((M / 16) + 3) / 4 * 4,
                pargs, num_persistent, dshmem, stream);
        }
        TORCH_CHECK(launch_err == cudaSuccess,
                    "persistent grouped dim1 launch failed: ",
                    cudaGetErrorString(launch_err));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_group_quantize_dim1_launch failed: ", cudaGetErrorString(err));

    // Build output views (zero-allocation, just views/narrows of pre-allocated tensors)
    std::vector<torch::Tensor> fp4_row_list(G), sc_row_list(G);
    std::vector<torch::Tensor> fp4_col_list(G), sc_col_list(G);
    std::vector<torch::Tensor> sc_row_u8_list(G), sc_col_u8_list(G);
    int64_t col_offset = 0;
    for (int g = 0; g < G; ++g) {
        const int64_t N_g = col_split_sections[g];
        fp4_row_list[g] = fp4_row_full.narrow(1, col_offset / 2, N_g / 2);
        sc_row_list[g] = sc_row_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_row_u8_list[g] = sc_row_allocs[g];
        fp4_col_list[g] = fp4_col_full.narrow(0, col_offset, N_g).view(torch::kFloat4_e2m1fn_x2);
        sc_col_list[g] = sc_col_allocs[g].view(torch::kFloat8_e4m3fn);
        sc_col_u8_list[g] = sc_col_allocs[g];
        col_offset += N_g;
    }

    torch::Tensor sc_row_cat, sc_col_cat;
    if (skip_cat) {
        // Graph-safe: skip torch::cat to avoid graph-pool allocations
        sc_row_cat = torch::empty({0}, torch::dtype(torch::kUInt8).device(input.device()));
        sc_col_cat = torch::empty({0}, torch::dtype(torch::kUInt8).device(input.device()));
    } else {
        sc_row_cat = torch::cat(sc_row_u8_list, 1).view(torch::kFloat8_e4m3fn);
        sc_col_cat = torch::cat(sc_col_u8_list, 0).view(torch::kFloat8_e4m3fn);
    }
    auto fp4_row_view = fp4_row_full.view(torch::kFloat4_e2m1fn_x2);
    auto fp4_col_view = fp4_col_full.view(torch::kFloat4_e2m1fn_x2);

    return std::make_tuple(fp4_row_list, sc_row_list, sg_per_group,
                           fp4_col_list, sc_col_list,
                           fp4_row_view, sc_row_cat,
                           fp4_col_view, sc_col_cat);
}
//
// Two paths:
//   FUSED: single-pass grid-barrier kernel (when grid fits on GPU)
//   FALLBACK: inv_rms → bf16 transform → pipelined amax → quantize with pre-computed amax
// ═══════════════════════════════════════════════════════════════════

// Helper: apply rmsnorm + optional silu, output bf16 + compute amax
// This is used in the fallback 2-pass path
template <bool WITH_SILU, int BLOCK_SIZE = 256>
__global__ void transform_and_amax_kernel(
    const __nv_bfloat16* __restrict__ x,       // (M, K) input
    const __nv_bfloat16* __restrict__ gamma,    // (K,) rmsnorm weight
    const float* __restrict__ inv_rms,          // (M,) pre-computed
    __nv_bfloat16* __restrict__ out,            // (M, K) transformed output
    float* __restrict__ global_amax,            // scalar (atomic max target)
    float* __restrict__ sg_out,                 // scalar
    int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* row_x = x + (int64_t)row * cols;
    __nv_bfloat16* row_out = out + (int64_t)row * cols;
    float row_inv_rms = inv_rms[row];
    float thread_max = 0.0f;

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_x[i]);
        float g = __bfloat162float(gamma[i]);
        float transformed = val * row_inv_rms * g;
        if constexpr (WITH_SILU) {
            transformed = transformed / (1.0f + expf(-transformed));
        }
        row_out[i] = __float2bfloat16_rn(transformed);
        thread_max = fmaxf(thread_max, fabsf(transformed));
    }

    // Warp reduce
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));

    // Block reduce
    __shared__ float warp_max[BLOCK_SIZE / 32];
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) warp_max[wid] = thread_max;
    __syncthreads();

    if (wid == 0) {
        thread_max = (lane < BLOCK_SIZE / 32) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = (BLOCK_SIZE / 32) / 2; mask > 0; mask >>= 1)
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    // Atomic max to global (skip if no amax output needed)
    if (threadIdx.x == 0 && thread_max > 0.0f && global_amax != nullptr) {
        unsigned int* p = reinterpret_cast<unsigned int*>(global_amax);
        unsigned int old = *p;
        unsigned int want = __float_as_uint(thread_max);
        while (want > old) {
            old = atomicCAS(p, old, want);
        }
    }
}

// Helper: compute sg from amax (launched as 1 thread)
__global__ void compute_sg_from_amax(const float* __restrict__ amax, float* __restrict__ sg) {
    *sg = *amax / 2688.0f;
}

static void split_fp4_row_groups_from_full(
    const torch::Tensor& fp4_row_full,
    torch::Tensor out1_fp4,
    torch::Tensor out2_fp4,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    const size_t src_pitch = static_cast<size_t>(H);
    const size_t dst_pitch = static_cast<size_t>(H / 2);
    const size_t copy_width = dst_pitch;
    const size_t copy_height = static_cast<size_t>(M);

    const auto* src_base = reinterpret_cast<const char*>(fp4_row_full.data_ptr());
    auto* dst1 = reinterpret_cast<char*>(out1_fp4.data_ptr());
    auto* dst2 = reinterpret_cast<char*>(out2_fp4.data_ptr());

    auto err = cudaMemcpy2DAsync(
        dst1, dst_pitch,
        src_base, src_pitch,
        copy_width, copy_height,
        cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(err == cudaSuccess,
                "split_fp4_row_groups_from_full(first) failed: ",
                cudaGetErrorString(err));

    err = cudaMemcpy2DAsync(
        dst2, dst_pitch,
        src_base + copy_width, src_pitch,
        copy_width, copy_height,
        cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(err == cudaSuccess,
                "split_fp4_row_groups_from_full(second) failed: ",
                cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor>  // inv_rms (for backward) + scratch keepalive
tk_fused_norm_quantize_impl(
    torch::Tensor input,        // (M, K) bf16 raw pre-norm
    torch::Tensor gamma,        // (K,) bf16 rmsnorm weight
    float epsilon,
    bool with_silu,
    bool return_transpose,
    std::optional<torch::Tensor> inv_rms_opt
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                    input.dtype() == torch::kBFloat16 && input.dim() == 2,
                "input must be contiguous CUDA bf16 [M,K]");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous() &&
                    gamma.dtype() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be contiguous CUDA bf16 [K]");
    TORCH_CHECK(input.device() == gamma.device(),
                "input and gamma must be on one CUDA device");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0f,
                "epsilon must be finite and non-negative");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(gamma.size(0) == K, "gamma must match K dimension");

    auto device = input.device();
    const c10::cuda::CUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    V5QuantSequenceGuard sequence_guard(stream);
    using namespace tk_v3;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    auto inv_rms_tensor = inv_rms_opt.has_value()
        ? inv_rms_opt.value()
        : torch::empty({M}, opts_f32);
    TORCH_CHECK(inv_rms_tensor.is_cuda() && inv_rms_tensor.is_contiguous() &&
                    inv_rms_tensor.scalar_type() == torch::kFloat32 &&
                    inv_rms_tensor.dim() == 1 && inv_rms_tensor.size(0) == M,
                "inv_rms must be contiguous CUDA float32 [M]");
    TORCH_CHECK(inv_rms_tensor.device() == input.device(),
                "inv_rms must be on the input device");
    float* inv_rms_ptr = inv_rms_tensor.data_ptr<float>();

    if (!with_silu && return_transpose && use_barrier_free_norm_quant()) {
        // Compute inverse RMS and the BF16-rounded normalized amax without a
        // device-wide barrier, then quantize directly from the raw input.
        // The combined producer avoids a dense normalized BF16 buffer and an
        // otherwise redundant pass when inv_rms was not supplied by a caller.
        auto orig_amax = tk_prepare_rmsnorm_bf16_amax(
            input, gamma, inv_rms_tensor, epsilon,
            !inv_rms_opt.has_value(), stream);

        const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
        const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;
        auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
        auto row_sc = torch::empty({M, scale_stride}, opts_u8);
        auto col_fp4 = torch::empty({K, M / 2}, opts_u8);
        auto col_sc = torch::empty({K, scale_stride_t}, opts_u8);

        alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(
            tmap_in, input.data_ptr(), M, K,
            BUFF_DIM_Y, BUFF_DIM_X, K, 16);
        create_tma_2d(
            tmap_out, row_fp4.data_ptr(), M, K,
            BUFF_DIM_Y, BUFF_DIM_X, K, 4);
        create_tma_2d(
            tmap_out_t, col_fp4.data_ptr(), K, M,
            BUFF_DIM_X, BUFF_DIM_Y, M, 4);

        const float *amax = orig_amax.data_ptr<float>();
        launch_v2_rmsnorm_kernel<
            false, false, true, true, false, false, false, true>(
                tmap_in,
                tmap_out,
                tmap_out_t,
                reinterpret_cast<nvfp4_scale_t *>(row_sc.data_ptr()),
                reinterpret_cast<nvfp4_scale_t *>(col_sc.data_ptr()),
                amax,
                amax,
                inv_rms_ptr,
                reinterpret_cast<const tk_v3::IType *>(gamma.data_ptr()),
                M,
                K,
                scale_stride,
                scale_stride_t,
                stream);

        auto err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "barrier-free fused RMSNorm quantization failed: ",
            cudaGetErrorString(err));
        return std::make_tuple(
            row_fp4.view(torch::kFloat4_e2m1fn_x2),
            row_sc.view(torch::kFloat8_e4m3fn),
            col_fp4.view(torch::kFloat4_e2m1fn_x2),
            col_sc.view(torch::kFloat8_e4m3fn),
            orig_amax.narrow(0, 1, 1),
            inv_rms_tensor,
            orig_amax.narrow(0, 0, 1),
            orig_amax,
            orig_amax);
    }

    if (!inv_rms_opt.has_value()) {
        constexpr int BS = 256;
        inv_rms_kernel_ns::compute_inv_rms_kernel<BS><<<M, BS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            inv_rms_ptr, epsilon, M, K);
    }

    // Step 2: Persistent fused norm+quantize
    // Uses v5 persistent pattern: work-stealing loops for Phase 1 (amax) and
    // Phase 2 (quantize). RMSNorm is applied inline (2 muls/element, cheap).
    // No intermediate bf16 buffer needed.
    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;
    const int64_t scale_stride_r = ntk_r * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_u8) : torch::empty({0}, opts_u8);

    // Keep persistent-kernel counters per invocation. A process-global counter
    // can be reset by a later launch while an earlier launch is still draining
    // under normal async trainer launch pressure.
    auto amax_buf = torch::empty({2}, opts_f32);      // [amax, sg]
    auto sync_buf = torch::empty({4}, opts_i32);      // [wc1, wc2, done, ready]
    float* s_amax_buf = amax_buf.data_ptr<float>();
    auto* s_sync_buf = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(s_amax_buf, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(s_sync_buf, 0, 4 * sizeof(unsigned int), stream);

    // Determine persistent block count
    static int s_norm_quant_max_bps = 0;
    int dshmem = tk_v5::persistent_norm_quant_smem_size(return_transpose);
    if (s_norm_quant_max_bps == 0) {
        auto kern = return_transpose ?
            (with_silu ? (void*)tk_v5::persistent_norm_quantize_kernel<true, true>
                       : (void*)tk_v5::persistent_norm_quantize_kernel<false, true>)
          : (with_silu ? (void*)tk_v5::persistent_norm_quantize_kernel<true, false>
                       : (void*)tk_v5::persistent_norm_quantize_kernel<false, false>);
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        int max_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kern, V3_THREADS, dshmem);
        s_norm_quant_max_bps = max_blocks;
    }
    const auto& ci = get_cached_info();
    int num_persistent = std::min(
        s_norm_quant_max_bps * persistent_launch_sms(ci), total_tiles);

    // TMA maps
    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_norm_out{};
    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_norm_out, input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
    if (return_transpose)
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    // Scale TMA maps: view [ntm, ntk, 512] as 2D using BF16 type
    // Max TMA box = 512 bytes, so use shmemX=256 BF16 (512 bytes) per tile_k block
    // Each chunk does 2 TMA stores (one per tile_k × 512-byte block)
    {
        const int64_t sc_row_x_bf16 = ntk_r * 256;  // ntk_r*512 bytes / 2 = BF16 elements
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    }
    if (return_transpose) {
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    tk_v5::PersistentArgs p_args;
    p_args.work_counter_phase1 = s_sync_buf;
    p_args.work_counter_phase2 = s_sync_buf + 1;
    p_args.global_amax = s_amax_buf;
    p_args.done_counter = s_sync_buf + 2;
    p_args.ready_flag = s_sync_buf + 3;
    p_args.tiles_X = tiles_X;
    p_args.tiles_Y = tiles_Y;
    p_args.total_tiles = total_tiles;
    p_args.num_persistent = num_persistent;
    p_args.sg_output = s_amax_buf + 1;
    p_args.col_scales_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;
    p_args.col_scale_stride = 0;
    p_args.swizzle_scales = true;

    #define LAUNCH_PERSISTENT_NORM_QUANT(SILU_, TR_) \
        tk_v5::persistent_norm_quantize_kernel<SILU_, TR_><<<num_persistent, V3_THREADS, dshmem, stream>>>( \
            tmap_in, tmap_out, tmap_out_t, tmap_norm_out, tmap_sc_row, tmap_sc_col, \
            reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()), \
            inv_rms_ptr, \
            reinterpret_cast<const tk_v3::IType*>(gamma.data_ptr()), \
            M, K, scale_stride_r, p_args, false)

    if (with_silu && return_transpose)  { LAUNCH_PERSISTENT_NORM_QUANT(true,  true);  }
    else if (with_silu)                 { LAUNCH_PERSISTENT_NORM_QUANT(true,  false); }
    else if (return_transpose)          { LAUNCH_PERSISTENT_NORM_QUANT(false, true);  }
    else                                { LAUNCH_PERSISTENT_NORM_QUANT(false, false); }
    #undef LAUNCH_PERSISTENT_NORM_QUANT

    // Read back sg from device buffer
    auto sg_tensor = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(sg_tensor.data_ptr<float>(), s_amax_buf + 1,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);
    auto amax_tensor = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(amax_tensor.data_ptr<float>(), s_amax_buf,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto fp4_row = row_fp4.view(torch::kFloat4_e2m1fn_x2);
    auto sc_row  = row_sc.view(torch::kFloat8_e4m3fn);
	auto fp4_col = return_transpose ? col_fp4.view(torch::kFloat4_e2m1fn_x2) : col_fp4;
		auto sc_col  = return_transpose ? col_sc.view(torch::kFloat8_e4m3fn) : col_sc;
		return std::make_tuple(fp4_row, sc_row, fp4_col, sc_col,
	                           sg_tensor, inv_rms_tensor, amax_tensor,
		                           amax_buf, sync_buf);
		}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_fused_norm_quantize(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool with_silu,
    bool return_transpose
) {
    return tk_fused_norm_quantize_impl(
        input, gamma, epsilon, with_silu, return_transpose, std::nullopt);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_fused_norm_quantize_from_inv_rms(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    bool with_silu,
    bool return_transpose
) {
    return tk_fused_norm_quantize_impl(
        input, gamma, 0.0f, with_silu, return_transpose, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_fused_norm_quantize_from_row_rms_partial(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor row_rms_partial,
    double epsilon,
    bool with_silu,
    bool return_transpose
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                    input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be contiguous CUDA bf16 [M,K]");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous() &&
                    gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1,
                "gamma must be contiguous CUDA bf16 [K]");
    TORCH_CHECK(row_rms_partial.is_cuda() && row_rms_partial.is_contiguous() &&
                    row_rms_partial.scalar_type() == torch::kFloat32 &&
                    row_rms_partial.dim() == 2,
                "row_rms_partial must be contiguous CUDA float32 [M,K/32]");
    TORCH_CHECK(input.device() == gamma.device() &&
                    input.device() == row_rms_partial.device(),
                "input, gamma, and row_rms_partial must be on one CUDA device");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0,
                "M and K must be multiples of 128");
    TORCH_CHECK(K <= 4096,
                "row_rms_partial warp reducer supports K <= 4096");
    TORCH_CHECK(gamma.size(0) == K, "gamma must match K dimension");
    TORCH_CHECK(row_rms_partial.size(0) == M &&
                    row_rms_partial.size(1) == K / 32,
                "row_rms_partial shape must be [M,K/32]");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0,
                "epsilon must be finite and non-negative");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto inv_rms = torch::empty(
        {M}, torch::dtype(torch::kFloat32).device(input.device()));
    c1_rms_reduce::row_rms_reduce_entrypoint(
        row_rms_partial, inv_rms, K, epsilon);
    return tk_fused_norm_quantize_impl(
        input, gamma, 0.0f, with_silu, return_transpose, inv_rms);
}

template <bool ENCODE_CENTRIC>
static void launch_fused_norm_row_rht_persistent(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor inv_rms,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    cudaStream_t stream) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    using namespace tk_v3;

    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128;
    const int64_t ntk_r = K / 64;
    const int64_t ntm_c = K / 128;
    const int64_t ntk_c = M / 64;
    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;

    static int s_dshmem = -1;
    static int s_max_bps = -1;
    if (s_dshmem < 0) {
        s_dshmem = tk_v5::persistent_norm_quant_smem_size(true);
        auto kernel = tk_v5::persistent_norm_row_rht_quantize_kernel<
            true, true, ENCODE_CENTRIC>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, s_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&s_max_bps, kernel, V3_THREADS, s_dshmem);
    }

    const auto& ci = get_cached_info();
    int num_persistent = s_max_bps * persistent_launch_sms(ci);
    if (num_persistent > total_tiles) {
        num_persistent = total_tiles;
    }
    TORCH_CHECK(num_persistent > 0, "norm row-RHT persistent launch has zero resident CTAs");

    cudaMemsetAsync(amaxes.data_ptr<float>(), 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(sync_buf.data_ptr<int>(), 0, 4 * sizeof(unsigned int), stream);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    auto* sync_ptr = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int>());
    float* amax_ptr = amaxes.data_ptr<float>();

    tk_v5::PersistentArgs pargs;
    memset(&pargs, 0, sizeof(pargs));
    pargs.work_counter_phase1 = sync_ptr;
    pargs.work_counter_phase2 = sync_ptr + 1;
    pargs.global_amax = amax_ptr;
    pargs.done_counter = sync_ptr + 2;
    pargs.ready_flag = sync_ptr + 3;
    pargs.tiles_X = tiles_X;
    pargs.tiles_Y = tiles_Y;
    pargs.total_tiles = total_tiles;
    pargs.num_persistent = num_persistent;
    pargs.sg_output = sgs.data_ptr<float>();

    const dim3 grid(num_persistent);
    tk_v5::persistent_norm_row_rht_quantize_kernel<
        true, true, ENCODE_CENTRIC>
        <<<grid, V3_THREADS, s_dshmem, stream>>>(
            tmap_in,
            tmap_out,
            tmap_out_t,
            tmap_sc_row,
            tmap_sc_col,
            inv_rms.data_ptr<float>(),
            reinterpret_cast<const tk_v3::IType*>(gamma.data_ptr()),
            M,
            K,
            scale_stride,
            pargs,
            amax_ptr + 1);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor>
tk_fused_norm_quantize_row_rht_persistent(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool encode_centric) {
    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    auto device = input.device();
    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto inv_rms_tensor = torch::empty({M}, opts_f32);
    {
        constexpr int BS = 256;
        inv_rms_kernel_ns::compute_inv_rms_kernel<BS><<<M, BS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            inv_rms_tensor.data_ptr<float>(), epsilon, M, K);
    }

    const int64_t scale_stride = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_u8);
    auto col_sc = torch::empty({K, scale_stride_t}, opts_u8);
    auto sgs = torch::empty({2}, opts_f32);
    auto amaxes = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);

    if (encode_centric) {
        launch_fused_norm_row_rht_persistent<true>(
            input, gamma, inv_rms_tensor, row_fp4, row_sc, col_fp4, col_sc,
            sgs, amaxes, sync_buf, stream);
    } else {
        launch_fused_norm_row_rht_persistent<false>(
            input, gamma, inv_rms_tensor, row_fp4, row_sc, col_fp4, col_sc,
            sgs, amaxes, sync_buf, stream);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_fused_norm_quantize_row_rht_persistent failed: ",
                cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4.view(torch::kFloat4_e2m1fn_x2),
        row_sc.view(torch::kFloat8_e4m3fn),
        col_fp4.view(torch::kFloat4_e2m1fn_x2),
        col_sc.view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 0, 1),
        sgs.narrow(0, 1, 1),
        inv_rms_tensor);
}

	std::tuple<torch::Tensor, torch::Tensor,
	           torch::Tensor, torch::Tensor,
	           torch::Tensor, torch::Tensor,
	           torch::Tensor>
tk_fused_norm_quantize_opt(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(gamma.is_cuda() && gamma.is_contiguous(), "gamma must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2, "input must be 2D BF16");
    TORCH_CHECK(gamma.scalar_type() == torch::kBFloat16 && gamma.dim() == 1, "gamma must be 1D BF16");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(M % 16 == 0 && K % 16 == 0, "M and K must be multiples of 16");
    TORCH_CHECK(gamma.size(0) == K, "gamma must match K dimension");
    TORCH_CHECK(return_transpose, "tk_fused_norm_quantize_opt currently requires return_transpose=True");
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    V5QuantSequenceGuard sequence_guard(stream);

    for (char &c : rht_axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    TORCH_CHECK(
        rht_axes == "none" || rht_axes == "row" || rht_axes == "col" || rht_axes == "both" ||
        rht_axes == "rowcol" || rht_axes == "row_col",
        "Unsupported regular TK RMSNorm RHT axes: ", rht_axes);
    TORCH_CHECK(row_rht && !col_rht,
                "tk_fused_norm_quantize_opt currently supports the regular-TK QKV row-RHT contract only");
    TORCH_CHECK(!with_random_sign_mask,
                "tk_fused_norm_quantize_opt does not yet fuse random-sign RHT; disable NVFP4_RHT_RANDOM_SIGNS");
    if (!data_stochastic_rounding && !scale_stochastic_rounding) {
        return tk_fused_norm_quantize_row_rht_persistent(
            input, gamma, epsilon, encode_centric);
    }

    auto device = input.device();
    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto rng_state = tk_make_advancing_rng_state(
        input, rng_seed, rng_subsequence, stream);
    const auto *rng_state_ptr = reinterpret_cast<const size_t *>(
        rng_state.data_ptr<int64_t>());

    auto inv_rms_tensor = torch::empty({M}, opts_f32);
    {
        constexpr int BS = 256;
        inv_rms_kernel_ns::compute_inv_rms_kernel<BS><<<M, BS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            inv_rms_tensor.data_ptr<float>(), epsilon, M, K);
    }

    auto scans = tk_scan_rmsnorm_rht16_and_orig_amax(
        input, gamma, inv_rms_tensor, true, false, rng_seed, rng_subsequence, stream);
    torch::Tensor row_amax = std::get<0>(scans);
    torch::Tensor col_amax = std::get<1>(scans);

    const int64_t scale_stride   = ((K / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;
    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto row_sc  = torch::empty({M, scale_stride}, opts_u8);
    auto col_fp4 = torch::empty({K, M / 2}, opts_u8);
    auto col_sc  = torch::empty({K, scale_stride_t}, opts_u8);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_in,  input.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, BUFF_DIM_Y, BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, BUFF_DIM_X, BUFF_DIM_Y, M, 4);

    auto *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
    auto *sc_t_ptr = reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr());
    const float *amax_r = reinterpret_cast<const float*>(row_amax.narrow(0, 0, 1).data_ptr());
    const float *amax_c = reinterpret_cast<const float*>(col_amax.narrow(0, 0, 1).data_ptr());
    const auto *gamma_ptr = reinterpret_cast<const tk_v3::IType*>(gamma.data_ptr());

#define TK_LAUNCH_V2_RMSNORM(SR_FLAG, SCALE_SR_FLAG, ENCODE_FLAG)                           \
    launch_v2_rmsnorm_kernel<SR_FLAG, false, true, ENCODE_FLAG, SCALE_SR_FLAG, true, false, true>( \
        tmap_in, tmap_out, tmap_out_t, sc_ptr, sc_t_ptr, amax_r, amax_c,                    \
        inv_rms_tensor.data_ptr<float>(), gamma_ptr, M, K, scale_stride, scale_stride_t,     \
        stream, rng_state_ptr)

    if (data_stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) { TK_LAUNCH_V2_RMSNORM(true, true, true); }
            else { TK_LAUNCH_V2_RMSNORM(true, true, false); }
        } else if (encode_centric) {
            TK_LAUNCH_V2_RMSNORM(true, false, true);
        } else {
            TK_LAUNCH_V2_RMSNORM(true, false, false);
        }
    } else if (scale_stochastic_rounding) {
        if (encode_centric) { TK_LAUNCH_V2_RMSNORM(false, true, true); }
        else { TK_LAUNCH_V2_RMSNORM(false, true, false); }
    } else if (encode_centric) {
        TK_LAUNCH_V2_RMSNORM(false, false, true);
    } else {
        TK_LAUNCH_V2_RMSNORM(false, false, false);
    }

#undef TK_LAUNCH_V2_RMSNORM

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_fused_norm_quantize_opt failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        row_fp4.view(torch::kFloat4_e2m1fn_x2),
        row_sc.view(torch::kFloat8_e4m3fn),
        col_fp4.view(torch::kFloat4_e2m1fn_x2),
        col_sc.view(torch::kFloat8_e4m3fn),
        row_amax.narrow(0, 1, 1),
        col_amax.narrow(0, 1, 1),
        inv_rms_tensor
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_fused_norm_quantize_with_output(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon,
    bool with_silu,
    bool return_transpose
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16, "input must be CUDA bf16");
    TORCH_CHECK(gamma.is_cuda() && gamma.dtype() == torch::kBFloat16, "gamma must be CUDA bf16");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(gamma.size(0) == K, "gamma must match K dimension");

    auto device = input.device();
    const c10::cuda::CUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    V5QuantSequenceGuard sequence_guard(stream);
    using namespace tk_v3;

    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto inv_rms_tensor = torch::empty({M}, opts_f32);
    float* inv_rms_ptr = inv_rms_tensor.data_ptr<float>();
    {
        constexpr int BS = 256;
        inv_rms_kernel_ns::compute_inv_rms_kernel<BS><<<M, BS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            inv_rms_ptr, epsilon, M, K);
    }

    const int tiles_X = K / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const int64_t ntm_r = M / 128, ntk_r = K / 64;
    const int64_t ntm_c = K / 128, ntk_c = M / 64;
    const int64_t scale_stride_r = ntk_r * 4;

    auto normed = torch::empty({M, K}, opts_bf16);
    auto row_fp4 = torch::empty({M, K / 2}, opts_u8);
    auto col_fp4 = return_transpose ? torch::empty({K, M / 2}, opts_u8) : torch::empty({0}, opts_u8);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto col_sc = return_transpose ? torch::empty({ntm_c, ntk_c, 512}, opts_u8) : torch::empty({0}, opts_u8);

    auto opts_i32 = torch::dtype(torch::kInt32).device(device);
    auto amax_buf = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);
    float* s_amax_buf = amax_buf.data_ptr<float>();
    auto* s_sync_buf = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(s_amax_buf, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(s_sync_buf, 0, 4 * sizeof(unsigned int), stream);

    static int s_norm_quant_out_max_bps = 0;
    int dshmem = tk_v5::persistent_norm_quant_smem_size(return_transpose);
    if (s_norm_quant_out_max_bps == 0) {
        auto kern = return_transpose ?
            (with_silu ? (void*)tk_v5::persistent_norm_quantize_kernel<true, true>
                       : (void*)tk_v5::persistent_norm_quantize_kernel<false, true>)
          : (with_silu ? (void*)tk_v5::persistent_norm_quantize_kernel<true, false>
                       : (void*)tk_v5::persistent_norm_quantize_kernel<false, false>);
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        int max_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kern, V3_THREADS, dshmem);
        s_norm_quant_out_max_bps = max_blocks;
    }
    const auto& ci = get_cached_info();
    int num_persistent = std::min(
        s_norm_quant_out_max_bps * persistent_launch_sms(ci), total_tiles);

    alignas(64) CUtensorMap tmap_in{}, tmap_out{}, tmap_out_t{}, tmap_norm_out{};
    alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
    create_tma_2d(tmap_in, input.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 4);
    create_tma_2d(tmap_norm_out, normed.data_ptr(), M, K, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, K, 16);
    if (return_transpose) {
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), K, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
    }

    {
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    }
    if (return_transpose) {
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    }

    tk_v5::PersistentArgs p_args;
    p_args.work_counter_phase1 = s_sync_buf;
    p_args.work_counter_phase2 = s_sync_buf + 1;
    p_args.global_amax = s_amax_buf;
    p_args.done_counter = s_sync_buf + 2;
    p_args.ready_flag = s_sync_buf + 3;
    p_args.tiles_X = tiles_X;
    p_args.tiles_Y = tiles_Y;
    p_args.total_tiles = total_tiles;
    p_args.num_persistent = num_persistent;
    p_args.sg_output = s_amax_buf + 1;
    p_args.col_scales_ptr = return_transpose ? reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()) : nullptr;
    p_args.col_scale_stride = 0;
    p_args.swizzle_scales = true;

    #define LAUNCH_PERSISTENT_NORM_QUANT_OUT(SILU_, TR_) \
        tk_v5::persistent_norm_quantize_kernel<SILU_, TR_><<<num_persistent, V3_THREADS, dshmem, stream>>>( \
            tmap_in, tmap_out, tmap_out_t, tmap_norm_out, tmap_sc_row, tmap_sc_col, \
            reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()), \
            inv_rms_ptr, \
            reinterpret_cast<const tk_v3::IType*>(gamma.data_ptr()), \
            M, K, scale_stride_r, p_args, true)

    if (with_silu && return_transpose) { LAUNCH_PERSISTENT_NORM_QUANT_OUT(true, true); }
    else if (with_silu) { LAUNCH_PERSISTENT_NORM_QUANT_OUT(true, false); }
    else if (return_transpose) { LAUNCH_PERSISTENT_NORM_QUANT_OUT(false, true); }
    else { LAUNCH_PERSISTENT_NORM_QUANT_OUT(false, false); }
    #undef LAUNCH_PERSISTENT_NORM_QUANT_OUT

    auto sg_tensor = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(sg_tensor.data_ptr<float>(), s_amax_buf + 1,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);
    auto amax_tensor = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(amax_tensor.data_ptr<float>(), s_amax_buf,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto fp4_row = row_fp4.view(torch::kFloat4_e2m1fn_x2);
    auto sc_row = row_sc.view(torch::kFloat8_e4m3fn);
    auto fp4_col = return_transpose ? col_fp4.view(torch::kFloat4_e2m1fn_x2) : col_fp4;
    auto sc_col = return_transpose ? col_sc.view(torch::kFloat8_e4m3fn) : col_sc;
    return std::make_tuple(normed, fp4_row, sc_row, fp4_col, sc_col,
                           sg_tensor, inv_rms_tensor, amax_tensor,
                           amax_buf, sync_buf);
}


// ═══════════════════════════════════════════════════════════════════
// Fused strided-SiLU + quantize
// Input: h13 (M, 2H) bf16. Applies silu(h1)*h3 → quantize to NVFP4.
// Output: same as tk_quantize_for_gemm but for (M, H).
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_silu_quantize_for_gemm(torch::Tensor h13, int64_t H) {
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);
    const int64_t M = h13.size(0);
    TORCH_CHECK(h13.size(1) == 2 * H, "h13 must have shape (M, 2*H)");
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = h13.device();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    // Common output setup
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride   = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    // SCALE_MAX is read-only after initialization, so it can remain process
    // global. Mutable amax/sync counters are allocated per invocation below.
    static float *s_const_amax_buf = nullptr;  // constant SCALE_MAX for phase2 path
    if (!s_const_amax_buf) {
        cudaMalloc(&s_const_amax_buf, 2 * sizeof(float));
        // Set constant SCALE_MAX = fp4_max * fp8_max = 6.0 * 448.0 = 2688.0
        float scale_max = 2688.0f;
        cudaMemcpy(s_const_amax_buf, &scale_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(s_const_amax_buf + 1, &scale_max, sizeof(float), cudaMemcpyHostToDevice);
    }

    const auto& ci = get_cached_info();

    // Cached shmem size and occupancy (computed once)
    static int s_dshmem = -1;
    static int s_np_max_bps = -1;
    static int s_p2_dshmem = -1;
    static int s_p2_max_bps = -1;
    if (s_dshmem < 0) {
        s_dshmem = fused_silu_quant::fused_silu_quant_smem_size<true>();
        cudaFuncSetAttribute(
            fused_silu_quant::fused_silu_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_np_max_bps,
            fused_silu_quant::fused_silu_quantize_kernel<true>,
            V3_THREADS, s_dshmem);
        // Phase2-only kernel init (single-pass, no amax scan)
        s_p2_dshmem = ci.v3_dshmem_t;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_p2_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_p2_max_bps,
            tk_v5::persistent_quantize_phase2_kernel<true, false>,
            V3_THREADS, s_p2_dshmem);
    }

    const int max_concurrent = s_np_max_bps * persistent_launch_sms(ci);
    const bool can_fuse = (total_tiles <= max_concurrent && s_np_max_bps > 0);

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    auto amax_buf = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);
    float *s_amax_buf = amax_buf.data_ptr<float>();
    auto *s_sync_buf = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());

    // Declare output tensors here, allocate in each branch for better pipelining
    torch::Tensor row_fp4, row_sc, col_fp4, col_sc;
    auto h_silu_keepalive = torch::empty({0}, h13.options());

    if (can_fuse) {
        // ─── NON-PERSISTENT PATH: single kernel, data stays in SMEM ───
        row_fp4 = torch::empty({M, H / 2}, opts_fp4);
        row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        col_fp4 = torch::empty({H, M / 2}, opts_fp4);
        col_sc  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        cudaMemsetAsync(s_amax_buf, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(s_sync_buf, 0, 4 * sizeof(unsigned int), stream);

        nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
        nvfp4_scale_t *sc_t_ptr = reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr());

        alignas(64) CUtensorMap tmap_h1{}, tmap_h3{}, tmap_out{}, tmap_out_t{};
        create_tma_2d(tmap_h1, h13.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        void* h3_ptr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h13.data_ptr()) + H * sizeof(__nv_bfloat16));
        create_tma_2d(tmap_h3, h3_ptr, M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        create_tma_2d(tmap_out, row_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out_t, col_fp4.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        const dim3 grid(tiles_X, tiles_Y);
        fused_silu_quant::fused_silu_quantize_kernel<true><<<grid, V3_THREADS, s_dshmem, stream>>>(
            tmap_h1, tmap_h3, tmap_out, tmap_out_t,
            sc_ptr, sc_t_ptr,
            s_amax_buf, s_amax_buf + 1,
            s_sync_buf, s_sync_buf + 1,
            M, H, scale_stride, scale_stride_t,
            total_tiles);
    } else {
        // ─── LARGE GRID PATH: silu + SINGLE-PASS phase2 quantize ───
        // Uses constant SCALE_MAX=2688.0 as global amax → no Phase 1 amax scan.
        // Halves GMEM reads vs the two-pass persistent_quantize_kernel.

        // Step 1: allocate silu output + launch silu kernel → GPU starts immediately
        auto h_silu = torch::empty({M, H}, h13.options());  // bf16
        h_silu_keepalive = h_silu;
        {
            const __nv_bfloat16* h13_ptr = reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr());
            __nv_bfloat16* out_ptr = reinterpret_cast<__nv_bfloat16*>(h_silu.data_ptr());
            const int64_t total_pairs = (int64_t)M * H / 2;
            const int threads = 256;
            int blocks = (int)((total_pairs + threads - 1) / threads);
            if (blocks > 65535) blocks = 65535;
            silu_strided_kernel<<<blocks, threads, 0, stream>>>(
                h13_ptr, out_ptr, M, H);
        }
        // GPU is now running silu kernel! Do remaining setup on CPU:

        // Step 2: allocate output tensors (CPU work, overlapped with GPU silu)
        row_fp4 = torch::empty({M, H / 2}, opts_fp4);
        row_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        col_fp4 = torch::empty({H, M / 2}, opts_fp4);
        col_sc  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        // Phase2 only needs work_counter reset (no amax/done/ready)
        cudaMemsetAsync(s_sync_buf, 0, sizeof(unsigned int), stream);

        nvfp4_scale_t *sc_ptr   = reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr());
        nvfp4_scale_t *sc_t_ptr = reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr());

        // Step 3: create TMA maps (CPU work, overlapped with GPU silu)
        alignas(64) CUtensorMap tmap_in_q{}, tmap_out_q{}, tmap_out_t_q{};
        create_tma_2d(tmap_in_q, h_silu.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_out_q, row_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out_t_q, col_fp4.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        // Step 4: launch single-pass phase2 quantize (constant amax, no grid barrier)
        int num_persistent = s_p2_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::Phase2Args p2args;
        p2args.work_counter = s_sync_buf;
        p2args.global_amax  = s_const_amax_buf;  // constant SCALE_MAX = 2688.0
        p2args.tiles_X = tiles_X;
        p2args.tiles_Y = tiles_Y;
        p2args.total_tiles = total_tiles;
        p2args.sg_output = s_amax_buf + 1;  // sg = amax/2688 = 1.0

        const dim3 grid(num_persistent);
        tk_v5::persistent_quantize_phase2_kernel<true><<<grid, V3_THREADS, s_p2_dshmem, stream>>>(
            tmap_in_q, tmap_out_q, tmap_out_t_q, tmap_sc_row, tmap_sc_col,
            sc_ptr, M, H, scale_stride, p2args);
    }

    // Copy sg to output tensor
    auto sg_buf = torch::empty({1}, opts_f32);
    cudaMemcpyAsync(sg_buf.data_ptr<float>(), s_amax_buf + 1, sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc,
                           sg_buf, sg_buf,
                           amax_buf, sync_buf, h_silu_keepalive);
}

torch::Tensor
tk_silu_mul_split_bf16(
    torch::Tensor h1_raw,
    torch::Tensor h3
) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have identical shapes");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto out = torch::empty_like(h1_raw);
    tk_silu_split::launch_forward(
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, H, stream);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_mul_split_bf16 failed: ", cudaGetErrorString(err));
    return out;
}

std::tuple<torch::Tensor, torch::Tensor>
tk_silu_deriv_split_bf16(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(h3.size(0) == M && h3.size(1) == H);
    TORCH_CHECK(h1_raw.size(0) == M && h1_raw.size(1) == H);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto out1 = torch::empty_like(dh);
    auto out2 = torch::empty_like(dh);
    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out2.data_ptr()),
        M, H, stream);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_split_bf16 failed: ", cudaGetErrorString(err));
    return std::make_tuple(out1, out2);
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_silu_quantize_split_for_gemm(
    torch::Tensor h1_raw,
    torch::Tensor h3
) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have identical shapes");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = h1_raw.device();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    const auto& ci = get_cached_info();
    static int s_split_fwd_fused_dshmem = -1;
    static int s_split_fwd_fused_max_bps = -1;
    static int s_split_fwd_p2_dshmem = -1;
    static int s_split_fwd_p2_max_bps = -1;
    if (s_split_fwd_fused_dshmem < 0) {
        s_split_fwd_fused_dshmem = fused_silu_quant::fused_silu_quant_smem_size<true>();
        cudaFuncSetAttribute(
            fused_silu_quant::fused_silu_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_split_fwd_fused_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_split_fwd_fused_max_bps,
            fused_silu_quant::fused_silu_quantize_kernel<true>,
            V3_THREADS, s_split_fwd_fused_dshmem);

        s_split_fwd_p2_dshmem = ci.v3_dshmem_t;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_split_fwd_p2_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_split_fwd_p2_max_bps,
            tk_v5::persistent_quantize_phase2_kernel<true>,
            V3_THREADS, s_split_fwd_p2_dshmem);
    }

    const int max_concurrent = s_split_fwd_fused_max_bps * persistent_launch_sms(ci);
    const bool can_fuse = (total_tiles <= max_concurrent && s_split_fwd_fused_max_bps > 0);

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    auto split_fwd_amax = torch::empty({1}, opts_f32);
    auto split_fwd_sync = torch::empty({2}, opts_i32);
    float* s_split_fwd_amax = split_fwd_amax.data_ptr<float>();
    auto* s_split_fwd_sync = reinterpret_cast<unsigned int*>(split_fwd_sync.data_ptr<int32_t>());

    auto row_fp4 = torch::empty({M, H / 2}, opts_fp4);
    auto row_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto col_fp4 = torch::empty({H, M / 2}, opts_fp4);
    auto col_sc = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
    auto sg_buf = torch::empty({1}, opts_f32);
    auto h_bf16_keepalive = torch::empty({0}, h1_raw.options());
    float* sg_ptr = sg_buf.data_ptr<float>();

    alignas(64) CUtensorMap tmap_h1{}, tmap_h3{}, tmap_out{}, tmap_out_t{};
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_out, row_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out_t, col_fp4.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    if (can_fuse) {
        cudaMemsetAsync(s_split_fwd_amax, 0, sizeof(float), stream);
        cudaMemsetAsync(s_split_fwd_sync, 0, 2 * sizeof(unsigned int), stream);

        const dim3 grid(tiles_X, tiles_Y);
        fused_silu_quant::fused_silu_quantize_kernel<true><<<grid, V3_THREADS, s_split_fwd_fused_dshmem, stream>>>(
            tmap_h1, tmap_h3, tmap_out, tmap_out_t,
            reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()),
            reinterpret_cast<nvfp4_scale_t*>(col_sc.data_ptr()),
            s_split_fwd_amax, sg_ptr,
            s_split_fwd_sync, s_split_fwd_sync + 1,
            M, H, scale_stride, scale_stride_t, total_tiles);
    } else {
        cudaMemsetAsync(s_split_fwd_amax, 0, sizeof(float), stream);
        cudaMemsetAsync(s_split_fwd_sync, 0, sizeof(unsigned int), stream);
        auto h_bf16 = torch::empty_like(h1_raw);
        h_bf16_keepalive = h_bf16;
        tk_silu_split::launch_forward_with_amax(
            reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(h_bf16.data_ptr()),
            s_split_fwd_amax, M, H, stream);

        alignas(64) CUtensorMap tmap_in{};
        create_tma_2d(tmap_in, h_bf16.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

        alignas(64) CUtensorMap tmap_sc_row{}, tmap_sc_col{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_row, row_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        create_tma_2d(tmap_sc_col, col_sc.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        tk_v5::Phase2Args args;
        memset(&args, 0, sizeof(args));
        args.work_counter = s_split_fwd_sync;
        args.global_amax = s_split_fwd_amax;
        args.tiles_X = tiles_X;
        args.tiles_Y = tiles_Y;
        args.total_tiles = total_tiles;
        args.sg_output = sg_ptr;

        int num_persistent = s_split_fwd_p2_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;
        const dim3 grid(num_persistent);

        tk_v5::persistent_quantize_phase2_kernel<true>
            <<<grid, V3_THREADS, s_split_fwd_p2_dshmem, stream>>>(
                tmap_in, tmap_out, tmap_out_t,
                tmap_sc_row, tmap_sc_col,
                reinterpret_cast<nvfp4_scale_t*>(row_sc.data_ptr()),
                M, H, scale_stride, args);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_quantize_split_for_gemm failed: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, sg_buf, sg_buf,
                           split_fwd_amax, split_fwd_sync, h_bf16_keepalive);
}

// ═══════════════════════════════════════════════════════════════════
// 6. Fused SiLU Derivative + Dual Quantize (BACKWARD)
// ═══════════════════════════════════════════════════════════════════

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_for_gemm_impl(
    torch::Tensor dh,
    torch::Tensor h13,
    int64_t H,
    bool use_delayed_scaling,
    torch::Tensor prev_amax,
    bool collect_current_amax
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);

    const int64_t M = dh.size(0);
    TORCH_CHECK(dh.size(1) == H);
    TORCH_CHECK(h13.size(0) == M && h13.size(1) == 2 * H);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = dh.device();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    // Common output setup
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    const auto& ci = get_cached_info();

    // Cached shmem size and occupancy for backward persistent kernel
    static int s_bwd_dshmem = -1;
    static int s_bwd_max_bps = -1;
    if (s_bwd_dshmem < 0) {
        s_bwd_dshmem = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_bwd_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_bwd_max_bps,
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            V3_THREADS, s_bwd_dshmem);
    }
    // Single-pass kernel (delayed scaling, no grid barrier)
    static int s_sp_dshmem = -1;
    static int s_sp_max_bps = -1;
    if (s_sp_dshmem < 0) {
        s_sp_dshmem = single_pass_silu_deriv_quant::single_pass_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            single_pass_silu_deriv_quant::persistent_silu_deriv_single_pass_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_sp_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_sp_max_bps,
            single_pass_silu_deriv_quant::persistent_silu_deriv_single_pass_kernel<true>,
            V3_THREADS, s_sp_dshmem);
    }

    const int max_concurrent = s_bwd_max_bps * persistent_launch_sms(ci);
    const bool can_fuse = (total_tiles <= max_concurrent && s_bwd_max_bps > 0);
    const bool has_prev_amax = prev_amax.defined() && prev_amax.numel() >= 2;
    if (use_delayed_scaling && has_prev_amax) {
        TORCH_CHECK(prev_amax.is_cuda() && prev_amax.is_contiguous());
        TORCH_CHECK(prev_amax.scalar_type() == torch::kFloat32);
    }

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);

    auto bwd_amax_buf = torch::empty({2}, opts_f32);
    auto bwd_sync_buf = torch::empty({4}, opts_i32);
    float *s_bwd_amax_buf = bwd_amax_buf.data_ptr<float>();
    auto *s_bwd_sync_buf = reinterpret_cast<unsigned int*>(bwd_sync_buf.data_ptr<int32_t>());

    torch::Tensor out1_fp4, out1_sc, out2_fp4, out2_sc;
    torch::Tensor out1_fp4_t, out1_sc_t, out2_fp4_t, out2_sc_t;
    auto sg_buf = torch::empty({2}, opts_f32);
    auto dh13_keepalive = torch::empty({0}, opts_bf16);
    auto fp4_row_full_keepalive = torch::empty({0}, opts_fp4);
    float *sg_ptr = sg_buf.data_ptr<float>();

    if (can_fuse) {
        // ─── PERSISTENT PATH: small M, all fits in concurrent CTAs ───
        out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out1_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out2_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
        out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        cudaMemsetAsync(s_bwd_amax_buf, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(s_bwd_sync_buf, 0, 4 * sizeof(unsigned int), stream);

        // TMA maps: inputs
        alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
        create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16);
        create_tma_2d(tmap_h1, h13.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        void* h3_ptr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h13.data_ptr()) + H * sizeof(__nv_bfloat16));
        create_tma_2d(tmap_h3, h3_ptr, M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);

        // TMA maps: row outputs
        alignas(64) CUtensorMap tmap_out1{}, tmap_out2{};
        create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);

        // TMA maps: col outputs
        alignas(64) CUtensorMap tmap_out1_t{}, tmap_out2_t{};
        create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
        create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        // TMA maps: scales
        alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        alignas(64) CUtensorMap tmap_sc_col1{}, tmap_sc_col2{};
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        int num_persistent = s_bwd_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = s_bwd_sync_buf;
        pargs.work_counter_phase2 = s_bwd_sync_buf + 1;
        pargs.global_amax  = s_bwd_amax_buf;
        pargs.done_counter = s_bwd_sync_buf + 2;
        pargs.ready_flag   = s_bwd_sync_buf + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;

        const dim3 grid(num_persistent);
        persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true><<<grid, V3_THREADS, s_bwd_dshmem, stream>>>(
            tmap_dh, tmap_h1, tmap_h3,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
            M, H, scale_stride, pargs, s_bwd_amax_buf + 1,
            nullptr, nullptr, 0, 0);
    } else if (use_delayed_scaling) {
        // ─── LARGE GRID: single-pass kernel (delayed scaling) ───
        cudaMemsetAsync(s_bwd_amax_buf, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(s_bwd_sync_buf, 0, sizeof(unsigned int), stream);

        out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out1_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out2_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
        out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);

        // S_enc = fp8_max * fp4_max / amax = 448 * 6 / 2688 = 1.0
        const float S_enc1 = 1.0f;
        const float S_enc2 = 1.0f;

        {
            alignas(64) CUtensorMap tmap_out1{}, tmap_out2{}, tmap_out1_t{}, tmap_out2_t{};
            create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
            create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
            create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
            create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

            alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{}, tmap_sc_col1{}, tmap_sc_col2{};
            const int64_t sc_row_x_bf16 = ntk_r * 256;
            const int64_t sc_col_x_bf16 = ntk_c * 256;
            create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
            create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
            create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
            create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

            single_pass_silu_deriv_quant::SinglePassArgs spargs;
            spargs.work_counter = s_bwd_sync_buf;
            spargs.amax_out1    = s_bwd_amax_buf;
            spargs.amax_out2    = s_bwd_amax_buf + 1;
            spargs.prev_amax    = has_prev_amax ? prev_amax.data_ptr<float>() : nullptr;
            spargs.tiles_X = tiles_X;
            spargs.tiles_Y = tiles_Y;
            spargs.total_tiles = total_tiles;
            spargs.sg_output = sg_ptr;
            spargs.collect_current_amax = collect_current_amax;

            int num_persistent = s_sp_max_bps * persistent_launch_sms(ci);
            if (num_persistent > total_tiles) num_persistent = total_tiles;
            const dim3 grid(num_persistent);

            const __nv_bfloat16* dh_p = reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr());
            const __nv_bfloat16* h13_p = reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr());

            single_pass_silu_deriv_quant::persistent_silu_deriv_single_pass_kernel<true>
                <<<grid, V3_THREADS, s_sp_dshmem, stream>>>(
                dh_p, h13_p,
                tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
                tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
                M, H, scale_stride, S_enc1, S_enc2, spargs);
        }
    } else {
        // ─── LARGE GRID: interleaved bf16 staging -> grouped dim1 quantize ───

        out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out1_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
        out2_sc  = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
        out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
        out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
        out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
        auto dh13_bf16 = torch::empty({M, 2 * H}, opts_bf16);
        dh13_keepalive = dh13_bf16;
        tk_silu_deriv_interleaved::launch(
            reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dh13_bf16.data_ptr()),
            M, H, stream);

        auto grouped_out = tk_group_quantize_dim1_for_gemm(dh13_bf16, {H, H});
        auto fp4_row_full = std::get<5>(grouped_out).view(torch::kUInt8);
        fp4_row_full_keepalive = fp4_row_full;

        split_fp4_row_groups_from_full(
            fp4_row_full, out1_fp4, out2_fp4, M, H, stream);
        out1_sc = std::get<1>(grouped_out)[0];
        out2_sc = std::get<1>(grouped_out)[1];
        out1_fp4_t = std::get<3>(grouped_out)[0];
        out2_fp4_t = std::get<3>(grouped_out)[1];
        out1_sc_t = std::get<4>(grouped_out)[0];
        out2_sc_t = std::get<4>(grouped_out)[1];
        sg_buf = std::get<2>(grouped_out);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        sg_buf.narrow(0, 0, 1), torch::zeros({1}, opts_f32),
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sg_buf.narrow(0, 1, 1), torch::zeros({1}, opts_f32),
        bwd_amax_buf, bwd_sync_buf, dh13_keepalive, fp4_row_full_keepalive
    );
}

auto tk_silu_deriv_quantize_for_gemm(
    torch::Tensor dh,
    torch::Tensor h13,
    int64_t H,
    bool use_delayed_scaling = false
) {
    return tk_silu_deriv_quantize_for_gemm_impl(
        dh, h13, H, use_delayed_scaling, torch::Tensor(), true
    );
}

auto tk_silu_deriv_quantize_for_gemm_delayed(
    torch::Tensor dh,
    torch::Tensor h13,
    int64_t H,
    torch::Tensor prev_amax,
    bool collect_current_amax = true
) {
    return tk_silu_deriv_quantize_for_gemm_impl(
        dh, h13, H, true, prev_amax, collect_current_amax
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_split_for_gemm_impl(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor prev_amax,
    bool use_delayed_scaling,
    bool collect_current_amax
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(h3.size(0) == M && h3.size(1) == H);
    TORCH_CHECK(h1_raw.size(0) == M && h1_raw.size(1) == H);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto device = dh.device();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;

    const auto& ci = get_cached_info();
    static int s_split_bwd_fused_dshmem = -1;
    static int s_split_bwd_fused_max_bps = -1;
    static int s_split_bwd_p2_dshmem = -1;
    static int s_split_bwd_p2_max_bps = -1;
    if (s_split_bwd_fused_dshmem < 0) {
        s_split_bwd_fused_dshmem = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_split_bwd_fused_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_split_bwd_fused_max_bps,
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            V3_THREADS, s_split_bwd_fused_dshmem);

        s_split_bwd_p2_dshmem = ci.v3_dshmem_t;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_split_bwd_p2_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_split_bwd_p2_max_bps,
            tk_v5::persistent_quantize_phase2_kernel<true>,
            V3_THREADS, s_split_bwd_p2_dshmem);
    }

    const int max_concurrent = s_split_bwd_fused_max_bps * persistent_launch_sms(ci);
    const bool can_fuse = (total_tiles <= max_concurrent && s_split_bwd_fused_max_bps > 0);
    const bool has_prev_amax = prev_amax.defined() && prev_amax.numel() >= 2;
    if (use_delayed_scaling) {
        TORCH_CHECK(has_prev_amax, "delayed split SiLU-deriv quantize requires prev_amax with 2 values");
        TORCH_CHECK(prev_amax.is_cuda() && prev_amax.is_contiguous());
        TORCH_CHECK(prev_amax.scalar_type() == torch::kFloat32);
    }

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
    auto out1_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
    auto out2_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
    auto out1_sc_t = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
    auto out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
    auto out2_sc_t = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
    auto sg_buf = torch::empty({2}, opts_f32);
    float* sg_ptr = sg_buf.data_ptr<float>();

    auto opts_i32 = torch::dtype(torch::kInt32).device(device);
    auto split_bwd_amax_buf = torch::empty({2}, opts_f32);
    auto split_bwd_sync_buf = torch::empty({4}, opts_i32);
    float* split_bwd_amax = split_bwd_amax_buf.data_ptr<float>();
    auto* split_bwd_sync = reinterpret_cast<unsigned int*>(split_bwd_sync_buf.data_ptr<int32_t>());
    cudaMemsetAsync(split_bwd_amax, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(split_bwd_sync, 0, 4 * sizeof(unsigned int), stream);
    auto dh1_keepalive = torch::empty({0}, dh.options());
    auto dh3_keepalive = torch::empty({0}, dh.options());

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

    alignas(64) CUtensorMap tmap_out1{}, tmap_out2{}, tmap_out1_t{}, tmap_out2_t{};
    create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
    create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{}, tmap_sc_col1{}, tmap_sc_col2{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    if (can_fuse) {
        int num_persistent = s_split_bwd_fused_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = split_bwd_sync;
        pargs.work_counter_phase2 = split_bwd_sync + 1;
        pargs.global_amax = split_bwd_amax;
        pargs.done_counter = split_bwd_sync + 2;
        pargs.ready_flag = split_bwd_sync + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;

        const dim3 grid(num_persistent);
        persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>
            <<<grid, V3_THREADS, s_split_bwd_fused_dshmem, stream>>>(
                tmap_dh, tmap_h1, tmap_h3,
                tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
                tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
                M, H, scale_stride, pargs, split_bwd_amax + 1,
                nullptr, nullptr, 0, 0);
    } else {
        auto dh1 = torch::empty_like(dh);
        auto dh3 = torch::empty_like(dh);
        dh1_keepalive = dh1;
        dh3_keepalive = dh3;
        if (use_delayed_scaling && !collect_current_amax) {
            tk_silu_split::launch_backward(
                reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh3.data_ptr()),
                M, H, stream);
        } else {
            tk_silu_split::launch_backward_with_amax(
                reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dh3.data_ptr()),
                split_bwd_amax, split_bwd_amax + 1,
                M, H, stream);
        }

        alignas(64) CUtensorMap tmap_in1{}, tmap_in2{};
        create_tma_2d(tmap_in1, dh1.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        create_tma_2d(tmap_in2, dh3.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

        int num_persistent = s_split_bwd_p2_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;
        const dim3 grid(num_persistent);

        tk_v5::Phase2Args args1;
        memset(&args1, 0, sizeof(args1));
        args1.work_counter = split_bwd_sync;
        args1.global_amax = use_delayed_scaling ? prev_amax.data_ptr<float>() : split_bwd_amax;
        args1.tiles_X = tiles_X;
        args1.tiles_Y = tiles_Y;
        args1.total_tiles = total_tiles;
        args1.sg_output = sg_ptr;
        tk_v5::persistent_quantize_phase2_kernel<true>
            <<<grid, V3_THREADS, s_split_bwd_p2_dshmem, stream>>>(
                tmap_in1, tmap_out1, tmap_out1_t,
                tmap_sc_row1, tmap_sc_col1,
                reinterpret_cast<nvfp4_scale_t*>(out1_sc.data_ptr()),
                M, H, scale_stride, args1);

        tk_v5::Phase2Args args2;
        memset(&args2, 0, sizeof(args2));
        args2.work_counter = split_bwd_sync;
        args2.global_amax = use_delayed_scaling ? (prev_amax.data_ptr<float>() + 1) : (split_bwd_amax + 1);
        args2.tiles_X = tiles_X;
        args2.tiles_Y = tiles_Y;
        args2.total_tiles = total_tiles;
        args2.sg_output = sg_ptr + 1;
        cudaMemsetAsync(split_bwd_sync, 0, sizeof(unsigned int), stream);
        tk_v5::persistent_quantize_phase2_kernel<true>
            <<<grid, V3_THREADS, s_split_bwd_p2_dshmem, stream>>>(
                tmap_in2, tmap_out2, tmap_out2_t,
                tmap_sc_row2, tmap_sc_col2,
                reinterpret_cast<nvfp4_scale_t*>(out2_sc.data_ptr()),
                M, H, scale_stride, args2);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_split_for_gemm failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        sg_buf.narrow(0, 0, 1), torch::zeros({1}, opts_f32),
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sg_buf.narrow(0, 1, 1), torch::zeros({1}, opts_f32),
        split_bwd_amax_buf, split_bwd_sync_buf, dh1_keepalive, dh3_keepalive
    );
}

auto tk_silu_deriv_quantize_split_for_gemm(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw
) {
    return tk_silu_deriv_quantize_split_for_gemm_impl(
        dh, h3, h1_raw, torch::Tensor(), false, true
    );
}

auto tk_silu_deriv_quantize_split_for_gemm_delayed(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor prev_amax,
    bool collect_current_amax = true
) {
    return tk_silu_deriv_quantize_split_for_gemm_impl(
        dh, h3, h1_raw, prev_amax, true, collect_current_amax
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_split_for_gemm_alloc(int64_t M, int64_t H, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;

    auto opts_fp4 = torch::dtype(torch::kFloat4_e2m1fn_x2).device(device);
    auto opts_fp8 = torch::dtype(torch::kFloat8_e4m3fn).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);

    auto out1_fp4 = torch::empty({M, H / 2}, opts_fp4);
    auto out1_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto out1_fp4_t = torch::empty({H, M / 2}, opts_fp4);
    auto out1_sc_t = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
    auto out2_fp4 = torch::empty({M, H / 2}, opts_fp4);
    auto out2_sc = torch::empty({ntm_r, ntk_r, 512}, opts_fp8);
    auto out2_fp4_t = torch::empty({H, M / 2}, opts_fp4);
    auto out2_sc_t = torch::empty({ntm_c, ntk_c, 512}, opts_fp8);
    auto sg_buf = torch::empty({2}, opts_f32);
    auto zero_buf = torch::zeros({1}, opts_f32);
    auto amax_buf = torch::empty({2}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);
    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3 = torch::empty({M, H}, opts_bf16);

    return std::make_tuple(
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sg_buf, zero_buf, amax_buf, sync_buf, dh1, dh3);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_split_for_gemm_launch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor out1_fp4,
    torch::Tensor out1_sc,
    torch::Tensor out1_fp4_t,
    torch::Tensor out1_sc_t,
    torch::Tensor out2_fp4,
    torch::Tensor out2_sc,
    torch::Tensor out2_fp4_t,
    torch::Tensor out2_sc_t,
    torch::Tensor sg_buf,
    torch::Tensor zero_buf,
    torch::Tensor amax_buf,
    torch::Tensor sync_buf,
    torch::Tensor dh1,
    torch::Tensor dh3
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(sg_buf.scalar_type() == torch::kFloat32 && sg_buf.numel() >= 2);
    TORCH_CHECK(amax_buf.scalar_type() == torch::kFloat32 && amax_buf.numel() >= 2);
    TORCH_CHECK(sync_buf.scalar_type() == torch::kInt32 && sync_buf.numel() >= 4);
    TORCH_CHECK(dh1.scalar_type() == torch::kBFloat16 && dh1.is_contiguous());
    TORCH_CHECK(dh3.scalar_type() == torch::kBFloat16 && dh3.is_contiguous());

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(h3.size(0) == M && h3.size(1) == H);
    TORCH_CHECK(h1_raw.size(0) == M && h1_raw.size(1) == H);
    TORCH_CHECK(dh1.size(0) == M && dh1.size(1) == H);
    TORCH_CHECK(dh3.size(0) == M && dh3.size(1) == H);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;

    const auto& ci = get_cached_info();
    static int s_split_prealloc_fused_dshmem = -1;
    static int s_split_prealloc_fused_max_bps = -1;
    static int s_split_prealloc_p2_dshmem = -1;
    static int s_split_prealloc_p2_max_bps = -1;
    if (s_split_prealloc_fused_dshmem < 0) {
        s_split_prealloc_fused_dshmem = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_split_prealloc_fused_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_split_prealloc_fused_max_bps,
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            V3_THREADS, s_split_prealloc_fused_dshmem);

        s_split_prealloc_p2_dshmem = ci.v3_dshmem_t;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_split_prealloc_p2_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_split_prealloc_p2_max_bps,
            tk_v5::persistent_quantize_phase2_kernel<true>,
            V3_THREADS, s_split_prealloc_p2_dshmem);
    }

    const int max_concurrent = s_split_prealloc_fused_max_bps * persistent_launch_sms(ci);
    const bool can_fuse = (total_tiles <= max_concurrent && s_split_prealloc_fused_max_bps > 0);

    float* split_bwd_amax = amax_buf.data_ptr<float>();
    unsigned int* split_bwd_sync = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    float* sg_ptr = sg_buf.data_ptr<float>();
    cudaMemsetAsync(split_bwd_amax, 0, 2 * sizeof(float), stream);
    cudaMemsetAsync(split_bwd_sync, 0, 4 * sizeof(unsigned int), stream);

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

    alignas(64) CUtensorMap tmap_out1{}, tmap_out2{}, tmap_out1_t{}, tmap_out2_t{};
    create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
    create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{}, tmap_sc_col1{}, tmap_sc_col2{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    if (can_fuse) {
        int num_persistent = s_split_prealloc_fused_max_bps * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = split_bwd_sync;
        pargs.work_counter_phase2 = split_bwd_sync + 1;
        pargs.global_amax = split_bwd_amax;
        pargs.done_counter = split_bwd_sync + 2;
        pargs.ready_flag = split_bwd_sync + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;

        const dim3 grid(num_persistent);
        persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>
            <<<grid, V3_THREADS, s_split_prealloc_fused_dshmem, stream>>>(
                tmap_dh, tmap_h1, tmap_h3,
                tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
                tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
                M, H, scale_stride, pargs, split_bwd_amax + 1,
                nullptr, nullptr, 0, 0);
    } else {
        tk_silu_split::launch_backward_with_amax(
            reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dh3.data_ptr()),
            split_bwd_amax, split_bwd_amax + 1,
            M, H, stream);

            alignas(64) CUtensorMap tmap_in1{}, tmap_in2{};
            create_tma_2d(tmap_in1, dh1.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            create_tma_2d(tmap_in2, dh3.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

            int num_persistent = s_split_prealloc_p2_max_bps * persistent_launch_sms(ci);
            if (num_persistent > total_tiles) num_persistent = total_tiles;
            const dim3 grid(num_persistent);

            tk_v5::Phase2Args args1;
            memset(&args1, 0, sizeof(args1));
            args1.work_counter = split_bwd_sync;
            args1.global_amax = split_bwd_amax;
            args1.tiles_X = tiles_X;
            args1.tiles_Y = tiles_Y;
            args1.total_tiles = total_tiles;
            args1.sg_output = sg_ptr;
            tk_v5::persistent_quantize_phase2_kernel<true>
                <<<grid, V3_THREADS, s_split_prealloc_p2_dshmem, stream>>>(
                    tmap_in1, tmap_out1, tmap_out1_t,
                    tmap_sc_row1, tmap_sc_col1,
                    reinterpret_cast<nvfp4_scale_t*>(out1_sc.data_ptr()),
                    M, H, scale_stride, args1);

            tk_v5::Phase2Args args2;
            memset(&args2, 0, sizeof(args2));
            args2.work_counter = split_bwd_sync;
            args2.global_amax = split_bwd_amax + 1;
            args2.tiles_X = tiles_X;
            args2.tiles_Y = tiles_Y;
            args2.total_tiles = total_tiles;
            args2.sg_output = sg_ptr + 1;
            cudaMemsetAsync(split_bwd_sync, 0, sizeof(unsigned int), stream);
            tk_v5::persistent_quantize_phase2_kernel<true>
                <<<grid, V3_THREADS, s_split_prealloc_p2_dshmem, stream>>>(
                    tmap_in2, tmap_out2, tmap_out2_t,
                    tmap_sc_row2, tmap_sc_col2,
                    reinterpret_cast<nvfp4_scale_t*>(out2_sc.data_ptr()),
                    M, H, scale_stride, args2);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_split_for_gemm_launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        sg_buf.narrow(0, 0, 1), zero_buf,
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sg_buf.narrow(0, 1, 1), zero_buf
    );
}

template <bool ROW_DATA_SR, bool COL_DATA_SR, bool ENCODE_CENTRIC, bool SCALE_SR>
static void launch_silu_split2_row_rht_persistent(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor out1_fp4,
    torch::Tensor out1_sc,
    torch::Tensor out1_fp4_t,
    torch::Tensor out1_sc_t,
    torch::Tensor out2_fp4,
    torch::Tensor out2_sc,
    torch::Tensor out2_fp4_t,
    torch::Tensor out2_sc_t,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    const uint64_t *rng_state_ptr,
    cudaStream_t stream
) {
    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;

    static int s_dshmem = -1;
    static int s_max_bps = -1;
    if (s_dshmem < 0) {
        s_dshmem = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        auto kernel =
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<
                true, true, ROW_DATA_SR, ENCODE_CENTRIC, SCALE_SR, COL_DATA_SR>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, s_dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&s_max_bps, kernel, V3_THREADS, s_dshmem);
    }

    const auto& ci = get_cached_info();
    int num_persistent = s_max_bps * persistent_launch_sms(ci);
    if (num_persistent > total_tiles) num_persistent = total_tiles;
    TORCH_CHECK(num_persistent > 0, "split2 row-RHT persistent launch has zero resident CTAs");

    cudaMemsetAsync(amaxes.data_ptr<float>(), 0, 4 * sizeof(float), stream);
    cudaMemsetAsync(sync_buf.data_ptr<int>(), 0, 4 * sizeof(unsigned int), stream);

    alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
    create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h1, h1_raw.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tmap_h3, h3.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);

    alignas(64) CUtensorMap tmap_out1{}, tmap_out2{}, tmap_out1_t{}, tmap_out2_t{};
    create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
    create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
    create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

    alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{}, tmap_sc_col1{}, tmap_sc_col2{};
    const int64_t sc_row_x_bf16 = ntk_r * 256;
    const int64_t sc_col_x_bf16 = ntk_c * 256;
    create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
    create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
    create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

    float* amax_ptr = amaxes.data_ptr<float>();
    auto* sync_ptr = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int>());

    tk_v5::PersistentArgs pargs;
    memset(&pargs, 0, sizeof(pargs));
    pargs.work_counter_phase1 = sync_ptr;
    pargs.work_counter_phase2 = sync_ptr + 1;
    pargs.global_amax = amax_ptr;
    pargs.done_counter = sync_ptr + 2;
    pargs.ready_flag = sync_ptr + 3;
    pargs.tiles_X = tiles_X;
    pargs.tiles_Y = tiles_Y;
    pargs.total_tiles = total_tiles;
    pargs.num_persistent = num_persistent;
    pargs.sg_output = sgs.data_ptr<float>();

    const dim3 grid(num_persistent);
    persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<
        true, true, ROW_DATA_SR, ENCODE_CENTRIC, SCALE_SR, COL_DATA_SR>
        <<<grid, V3_THREADS, s_dshmem, stream>>>(
            tmap_dh, tmap_h1, tmap_h3,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
            M, H, scale_stride, pargs,
            amax_ptr + 1, amax_ptr + 2, amax_ptr + 3,
            rng_seed, rng_subsequence, rng_state_ptr);
}

static void launch_silu_split2_row_rht_persistent_dispatch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor out1_fp4,
    torch::Tensor out1_sc,
    torch::Tensor out1_fp4_t,
    torch::Tensor out1_sc_t,
    torch::Tensor out2_fp4,
    torch::Tensor out2_sc,
    torch::Tensor out2_fp4_t,
    torch::Tensor out2_sc_t,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    bool row_data_stochastic_rounding,
    bool col_data_stochastic_rounding,
    bool encode_centric,
    bool scale_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    const uint64_t *rng_state_ptr,
    cudaStream_t stream
) {
#define TK_LAUNCH_SPLIT2_ROW_RHT(ROW_SR_FLAG, COL_SR_FLAG, ENCODE_FLAG, SCALE_SR_FLAG) \
    launch_silu_split2_row_rht_persistent<ROW_SR_FLAG, COL_SR_FLAG, ENCODE_FLAG, SCALE_SR_FLAG>( \
        dh, h3, h1_raw, out1_fp4, out1_sc, out1_fp4_t, out1_sc_t, \
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t, sgs, amaxes, sync_buf, \
        rng_seed, rng_subsequence, rng_state_ptr, stream)

    if (row_data_stochastic_rounding && col_data_stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) TK_LAUNCH_SPLIT2_ROW_RHT(true, true, true, true);
            else TK_LAUNCH_SPLIT2_ROW_RHT(true, true, false, true);
        } else if (encode_centric) {
            TK_LAUNCH_SPLIT2_ROW_RHT(true, true, true, false);
        } else {
            TK_LAUNCH_SPLIT2_ROW_RHT(true, true, false, false);
        }
    } else if (row_data_stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) TK_LAUNCH_SPLIT2_ROW_RHT(true, false, true, true);
            else TK_LAUNCH_SPLIT2_ROW_RHT(true, false, false, true);
        } else if (encode_centric) {
            TK_LAUNCH_SPLIT2_ROW_RHT(true, false, true, false);
        } else {
            TK_LAUNCH_SPLIT2_ROW_RHT(true, false, false, false);
        }
    } else if (col_data_stochastic_rounding) {
        if (scale_stochastic_rounding) {
            if (encode_centric) TK_LAUNCH_SPLIT2_ROW_RHT(false, true, true, true);
            else TK_LAUNCH_SPLIT2_ROW_RHT(false, true, false, true);
        } else if (encode_centric) {
            TK_LAUNCH_SPLIT2_ROW_RHT(false, true, true, false);
        } else {
            TK_LAUNCH_SPLIT2_ROW_RHT(false, true, false, false);
        }
    } else if (scale_stochastic_rounding) {
        if (encode_centric) TK_LAUNCH_SPLIT2_ROW_RHT(false, false, true, true);
        else TK_LAUNCH_SPLIT2_ROW_RHT(false, false, false, true);
    } else if (encode_centric) {
        TK_LAUNCH_SPLIT2_ROW_RHT(false, false, true, false);
    } else {
        TK_LAUNCH_SPLIT2_ROW_RHT(false, false, false, false);
    }

#undef TK_LAUNCH_SPLIT2_ROW_RHT
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_split_for_gemm_opt(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    std::string data_sr_axes
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h3.device() == dh.device() && h1_raw.device() == dh.device(),
                "split2 inputs must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(dh.device());
    const auto set_device_err = cudaSetDevice(dh.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before v5 split2 quantization: ",
                cudaGetErrorString(set_device_err));

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(h3.size(0) == M && h3.size(1) == H);
    TORCH_CHECK(h1_raw.size(0) == M && h1_raw.size(1) == H);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);
    TORCH_CHECK(return_transpose, "split2 opt producer currently requires return_transpose=True");

    for (char &c : rht_axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    TORCH_CHECK(
        rht_axes == "row",
        "split2 opt producer currently supports the row-RHT regular-TK contract only; got ", rht_axes);
    TORCH_CHECK(row_rht && !col_rht);
    TORCH_CHECK(!with_random_sign_mask, "split2 opt producer does not support random RHT sign masks yet");
    const auto sr_axes = resolve_data_sr_axes(
        data_stochastic_rounding, std::move(data_sr_axes), "v5 split2 opt");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    V5QuantSequenceGuard sequence_guard(stream);
    auto device = dh.device();
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);

    auto dh1 = torch::empty({M, H}, opts_bf16);
    auto dh3_out = torch::empty({M, H}, opts_bf16);
    auto amaxes = torch::empty({4}, opts_f32);
    auto sgs = torch::empty({4}, opts_f32);
    cudaMemsetAsync(amaxes.data_ptr<float>(), 0, 4 * sizeof(float), stream);

    tk_silu_split::launch_backward(
        reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h3.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(h1_raw.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(dh3_out.data_ptr()),
        M, H, stream);

    constexpr int threads = 256;
    const int64_t groups = M * (H / 16);
    const int blocks = static_cast<int>(std::min<int64_t>((groups + threads - 1) / threads, 65535));
    const int smem = 4 * pipelined_amax::WARPS * sizeof(float);
    dual_bf16_row_rht_orig_amax_kernel<<<blocks, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dh1.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dh3_out.data_ptr()),
        amaxes.data_ptr<float>(),
        M, H);
    compute_sg_kernel<<<1, 4, 0, stream>>>(amaxes.data_ptr<float>(), sgs.data_ptr<float>(), 4);

    auto q1_rng_state = (sr_axes.row || sr_axes.col || scale_stochastic_rounding)
        ? tk_make_advancing_rng_state(dh1, rng_seed, rng_subsequence, stream)
        : torch::Tensor();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> q1;
    if (sr_axes.row == sr_axes.col) {
        q1 = tk_quantize_transpose(
            dh1,
            amaxes.narrow(0, 0, 1),
            amaxes.narrow(0, 1, 1),
            true,
            sr_axes.row,
            encode_centric,
            scale_stochastic_rounding,
            true,
            false,
            true,
            q1_rng_state);
    } else {
        auto q1_row = tk_quantize_transpose(
            dh1,
            amaxes.narrow(0, 0, 1),
            amaxes.narrow(0, 1, 1),
            false,
            sr_axes.row,
            encode_centric,
            scale_stochastic_rounding,
            true,
            false,
            true,
            q1_rng_state);
        auto q1_col = tk_quantize_transpose(
            dh1,
            amaxes.narrow(0, 0, 1),
            amaxes.narrow(0, 1, 1),
            true,
            sr_axes.col,
            encode_centric,
            scale_stochastic_rounding,
            true,
            false,
            false,
            q1_rng_state);
        q1 = std::make_tuple(
            std::get<0>(q1_row), std::get<1>(q1_row),
            std::get<2>(q1_col), std::get<3>(q1_col));
    }
    auto q2_rng_state = (sr_axes.row || sr_axes.col || scale_stochastic_rounding)
        ? tk_make_advancing_rng_state(dh3_out, rng_seed, rng_subsequence, stream)
        : torch::Tensor();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> q2;
    if (sr_axes.row == sr_axes.col) {
        q2 = tk_quantize_transpose(
            dh3_out,
            amaxes.narrow(0, 2, 1),
            amaxes.narrow(0, 3, 1),
            true,
            sr_axes.row,
            encode_centric,
            scale_stochastic_rounding,
            true,
            false,
            true,
            q2_rng_state);
    } else {
        auto q2_row = tk_quantize_transpose(
            dh3_out,
            amaxes.narrow(0, 2, 1),
            amaxes.narrow(0, 3, 1),
            false,
            sr_axes.row,
            encode_centric,
            scale_stochastic_rounding,
            true,
            false,
            true,
            q2_rng_state);
        auto q2_col = tk_quantize_transpose(
            dh3_out,
            amaxes.narrow(0, 2, 1),
            amaxes.narrow(0, 3, 1),
            true,
            sr_axes.col,
            encode_centric,
            scale_stochastic_rounding,
            true,
            false,
            false,
            q2_rng_state);
        q2 = std::make_tuple(
            std::get<0>(q2_row), std::get<1>(q2_row),
            std::get<2>(q2_col), std::get<3>(q2_col));
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_split_for_gemm_opt failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        std::get<0>(q1).view(torch::kFloat4_e2m1fn_x2),
        std::get<1>(q1).view(torch::kFloat8_e4m3fn),
        std::get<2>(q1).view(torch::kFloat4_e2m1fn_x2),
        std::get<3>(q1).view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 0, 1),
        sgs.narrow(0, 1, 1),
        std::get<0>(q2).view(torch::kFloat4_e2m1fn_x2),
        std::get<1>(q2).view(torch::kFloat8_e4m3fn),
        std::get<2>(q2).view(torch::kFloat4_e2m1fn_x2),
        std::get<3>(q2).view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 2, 1),
        sgs.narrow(0, 3, 1),
        amaxes, dh1, dh3_out
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_split_for_gemm_opt_alloc(int64_t M, int64_t H, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    auto opts_u8 = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::dtype(torch::kInt32).device(device);

    auto out1_fp4 = torch::empty({M, H / 2}, opts_u8);
    auto out1_sc = torch::empty({M, scale_stride}, opts_u8);
    auto out1_fp4_t = torch::empty({H, M / 2}, opts_u8);
    auto out1_sc_t = torch::empty({H, scale_stride_t}, opts_u8);
    auto out2_fp4 = torch::empty({M, H / 2}, opts_u8);
    auto out2_sc = torch::empty({M, scale_stride}, opts_u8);
    auto out2_fp4_t = torch::empty({H, M / 2}, opts_u8);
    auto out2_sc_t = torch::empty({H, scale_stride_t}, opts_u8);
    auto sgs = torch::empty({4}, opts_f32);
    auto amaxes = torch::empty({4}, opts_f32);
    auto sync_buf = torch::empty({4}, opts_i32);

    return std::make_tuple(
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sgs, amaxes, sync_buf);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_split_for_gemm_opt_launch(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    bool return_transpose,
    bool encode_centric,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    std::string rht_axes,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    torch::Tensor out1_fp4,
    torch::Tensor out1_sc,
    torch::Tensor out1_fp4_t,
    torch::Tensor out1_sc_t,
    torch::Tensor out2_fp4,
    torch::Tensor out2_sc,
    torch::Tensor out2_fp4_t,
    torch::Tensor out2_sc_t,
    torch::Tensor sgs,
    torch::Tensor amaxes,
    torch::Tensor sync_buf,
    std::string data_sr_axes
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(sgs.is_cuda() && sgs.scalar_type() == torch::kFloat32 && sgs.numel() >= 4);
    TORCH_CHECK(amaxes.is_cuda() && amaxes.scalar_type() == torch::kFloat32 && amaxes.numel() >= 4);
    TORCH_CHECK(sync_buf.is_cuda() && sync_buf.scalar_type() == torch::kInt32 && sync_buf.numel() >= 4);
    TORCH_CHECK(h3.device() == dh.device() && h1_raw.device() == dh.device(),
                "split2 inputs must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(dh.device());
    const auto set_device_err = cudaSetDevice(dh.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before v5 split2 launch: ",
                cudaGetErrorString(set_device_err));

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(h3.size(0) == M && h3.size(1) == H);
    TORCH_CHECK(h1_raw.size(0) == M && h1_raw.size(1) == H);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);
    TORCH_CHECK(return_transpose, "split2 opt prealloc producer currently requires return_transpose=True");

    for (char &c : rht_axes) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (c == '-') c = '_';
    }
    const bool row_rht = (rht_axes == "row" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    const bool col_rht = (rht_axes == "col" || rht_axes == "both" || rht_axes == "rowcol" || rht_axes == "row_col");
    TORCH_CHECK(
        rht_axes == "row",
        "split2 opt prealloc producer currently supports the row-RHT regular-TK contract only; got ", rht_axes);
    TORCH_CHECK(row_rht && !col_rht);
    TORCH_CHECK(!with_random_sign_mask, "split2 opt prealloc producer does not support random RHT sign masks yet");
    const auto sr_axes = resolve_data_sr_axes(
        data_stochastic_rounding, std::move(data_sr_axes), "v5 split2 opt prealloc");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    V5QuantSequenceGuard sequence_guard(stream);
    auto rng_state = (sr_axes.row || sr_axes.col || scale_stochastic_rounding)
        ? tk_make_advancing_rng_state(dh, rng_seed, rng_subsequence, stream)
        : torch::Tensor();
    const auto *rng_state_ptr = rng_state.defined()
        ? reinterpret_cast<const uint64_t *>(rng_state.data_ptr<int64_t>())
        : nullptr;
    launch_silu_split2_row_rht_persistent_dispatch(
        dh, h3, h1_raw,
        out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
        out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
        sgs, amaxes, sync_buf,
        sr_axes.row, sr_axes.col, encode_centric, scale_stochastic_rounding,
        rng_seed, rng_subsequence, rng_state_ptr, stream);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_split_for_gemm_opt_launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(
        out1_fp4.view(torch::kFloat4_e2m1fn_x2),
        out1_sc.view(torch::kFloat8_e4m3fn),
        out1_fp4_t.view(torch::kFloat4_e2m1fn_x2),
        out1_sc_t.view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 0, 1),
        sgs.narrow(0, 1, 1),
        out2_fp4.view(torch::kFloat4_e2m1fn_x2),
        out2_sc.view(torch::kFloat8_e4m3fn),
        out2_fp4_t.view(torch::kFloat4_e2m1fn_x2),
        out2_sc_t.view(torch::kFloat8_e4m3fn),
        sgs.narrow(0, 2, 1),
        sgs.narrow(0, 3, 1)
    );
}

// ═══════════════════════════════════════════════════════════════════
// 6b. CUDA Graph-safe alloc/launch split for silu_deriv+quantize
// ═══════════════════════════════════════════════════════════════════

// _alloc: pre-create all output tensors + sync buffers. Call OUTSIDE graph capture.
// Returns: (out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
//           out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
//           sg_buf, amax_buf, sync_buf, psync_buf,
//           fp4_row_full, fp4_col_full, dh13_bf16, tma_host_buf, tma_dev_buf)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_for_gemm_alloc(int64_t M, int64_t H, torch::Device device) {
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0);
    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t n_total = 2 * H;

    auto opts_u8  = torch::dtype(torch::kUInt8).device(device);
    auto opts_f32 = torch::dtype(torch::kFloat32).device(device);
    auto opts_u32 = torch::dtype(torch::kInt32).device(device);
    auto opts_bf16 = torch::dtype(torch::kBFloat16).device(device);

    auto out1_fp4   = torch::empty({M, H / 2}, opts_u8);
    auto out1_sc    = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto out1_fp4_t = torch::empty({H, M / 2}, opts_u8);
    auto out1_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_u8);
    auto out2_fp4   = torch::empty({M, H / 2}, opts_u8);
    auto out2_sc    = torch::empty({ntm_r, ntk_r, 512}, opts_u8);
    auto out2_fp4_t = torch::empty({H, M / 2}, opts_u8);
    auto out2_sc_t  = torch::empty({ntm_c, ntk_c, 512}, opts_u8);
    auto sg_buf     = torch::empty({2}, opts_f32);
    auto amax_buf   = torch::empty({2}, opts_f32);
    auto sync_buf   = torch::empty({4}, opts_u32);
    auto psync_buf  = torch::empty({4}, opts_u32);
    auto fp4_row_full = torch::empty({M, n_total / 2}, opts_u8);
    auto fp4_col_full = torch::empty({n_total, M / 2}, opts_u8);
    auto dh13_bf16  = torch::empty({M, n_total}, opts_bf16);
    auto tma_host_buf = torch::empty(
        {(int64_t)(4 * (int64_t)sizeof(CUtensorMap))},
        torch::dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    auto tma_dev_buf = torch::empty(
        {(int64_t)(4 * (int64_t)sizeof(CUtensorMap))}, opts_u8);

    return std::make_tuple(out1_fp4, out1_sc, out1_fp4_t, out1_sc_t,
                           out2_fp4, out2_sc, out2_fp4_t, out2_sc_t,
                           sg_buf, amax_buf, sync_buf, psync_buf,
                           fp4_row_full, fp4_col_full, dh13_bf16,
                           tma_host_buf, tma_dev_buf);
}

// _launch: kernel-only dispatch — NO allocations, safe inside CUDA graph capture.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
tk_silu_deriv_quantize_for_gemm_launch(
    torch::Tensor dh, torch::Tensor h13, int64_t H,
    // Pre-allocated buffers from _alloc:
    torch::Tensor out1_fp4, torch::Tensor out1_sc,
    torch::Tensor out1_fp4_t, torch::Tensor out1_sc_t,
    torch::Tensor out2_fp4, torch::Tensor out2_sc,
    torch::Tensor out2_fp4_t, torch::Tensor out2_sc_t,
    torch::Tensor sg_buf, torch::Tensor amax_buf,
    torch::Tensor sync_buf, torch::Tensor psync_buf,
    torch::Tensor fp4_row_full, torch::Tensor fp4_col_full,
    torch::Tensor dh13_bf16,
    torch::Tensor tma_host_buf, torch::Tensor tma_dev_buf
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    const int64_t M = dh.size(0);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using namespace tk_v3;
    const int tiles_X = H / V3Config::CHUNK_DIM_X;
    const int tiles_Y = M / V3Config::CHUNK_DIM_Y;
    const int total_tiles = tiles_X * tiles_Y;

    const int64_t ntm_r = M / 128, ntk_r = H / 64;
    const int64_t ntm_c = H / 128, ntk_c = M / 64;
    const int64_t scale_stride = ((H / 16) + 3) / 4 * 4;
    const int64_t scale_stride_t = ((M / 16) + 3) / 4 * 4;

    float *amax_ptr = amax_buf.data_ptr<float>();
    float *sg_ptr = sg_buf.data_ptr<float>();
    unsigned int *sync_data = reinterpret_cast<unsigned int*>(sync_buf.data_ptr<int32_t>());
    const auto& ci = get_cached_info();

    static int s_bwd_dshmem_g = -1;
    static int s_bwd_max_bps_g = -1;
    static int s_p2_dshmem_g = -1;
    static int s_p2_max_bps_g = -1;
    if (s_bwd_dshmem_g < 0) {
        s_bwd_dshmem_g = persistent_silu_deriv_quant::fused_silu_deriv_quant_smem_size<true>();
        cudaFuncSetAttribute(
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_bwd_dshmem_g);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_bwd_max_bps_g,
            persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true>,
            V3_THREADS, s_bwd_dshmem_g);
        s_p2_dshmem_g = ci.v3_dshmem_t;
        cudaFuncSetAttribute(
            tk_v5::persistent_quantize_phase2_kernel<true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, s_p2_dshmem_g);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &s_p2_max_bps_g,
            tk_v5::persistent_quantize_phase2_kernel<true, false>,
            V3_THREADS, s_p2_dshmem_g);
    }

    const int max_concurrent = s_bwd_max_bps_g * persistent_launch_sms(ci);
    const bool can_fuse = (total_tiles <= max_concurrent && s_bwd_max_bps_g > 0);

    if (can_fuse) {
        // ─── PERSISTENT PATH ───
        cudaMemsetAsync(amax_ptr, 0, 2 * sizeof(float), stream);
        cudaMemsetAsync(sync_data, 0, 4 * sizeof(unsigned int), stream);

        alignas(64) CUtensorMap tmap_dh{}, tmap_h1{}, tmap_h3{};
        create_tma_2d(tmap_dh, dh.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 16);
        create_tma_2d(tmap_h1, h13.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);
        void* h3_ptr = reinterpret_cast<void*>(
            reinterpret_cast<char*>(h13.data_ptr()) + H * sizeof(__nv_bfloat16));
        create_tma_2d(tmap_h3, h3_ptr, M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, 2 * H, 16);

        alignas(64) CUtensorMap tmap_out1{}, tmap_out2{};
        create_tma_2d(tmap_out1, out1_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);
        create_tma_2d(tmap_out2, out2_fp4.data_ptr(), M, H, V3_BUFF_DIM_Y, V3_BUFF_DIM_X, H, 4);

        alignas(64) CUtensorMap tmap_out1_t{}, tmap_out2_t{};
        create_tma_2d(tmap_out1_t, out1_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);
        create_tma_2d(tmap_out2_t, out2_fp4_t.data_ptr(), H, M, V3_BUFF_DIM_X, V3_BUFF_DIM_Y, M, 4);

        alignas(64) CUtensorMap tmap_sc_row1{}, tmap_sc_row2{};
        const int64_t sc_row_x_bf16 = ntk_r * 256;
        create_tma_2d(tmap_sc_row1, out1_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        create_tma_2d(tmap_sc_row2, out2_sc.data_ptr(), ntm_r, sc_row_x_bf16, 1, 256, sc_row_x_bf16, 16);
        alignas(64) CUtensorMap tmap_sc_col1{}, tmap_sc_col2{};
        const int64_t sc_col_x_bf16 = ntk_c * 256;
        create_tma_2d(tmap_sc_col1, out1_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);
        create_tma_2d(tmap_sc_col2, out2_sc_t.data_ptr(), ntm_c, sc_col_x_bf16, 1, 256, sc_col_x_bf16, 16);

        int num_persistent = s_bwd_max_bps_g * persistent_launch_sms(ci);
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        tk_v5::PersistentArgs pargs;
        memset(&pargs, 0, sizeof(pargs));
        pargs.work_counter_phase1 = sync_data;
        pargs.work_counter_phase2 = sync_data + 1;
        pargs.global_amax  = amax_ptr;
        pargs.done_counter = sync_data + 2;
        pargs.ready_flag   = sync_data + 3;
        pargs.tiles_X = tiles_X;
        pargs.tiles_Y = tiles_Y;
        pargs.total_tiles = total_tiles;
        pargs.num_persistent = num_persistent;
        pargs.sg_output = sg_ptr;

        const dim3 grid(num_persistent);
        persistent_silu_deriv_quant::persistent_silu_deriv_quantize_kernel<true><<<grid, V3_THREADS, s_bwd_dshmem_g, stream>>>(
            tmap_dh, tmap_h1, tmap_h3,
            tmap_out1, tmap_out2, tmap_out1_t, tmap_out2_t,
            tmap_sc_row1, tmap_sc_row2, tmap_sc_col1, tmap_sc_col2,
            M, H, scale_stride, pargs, amax_ptr + 1,
            nullptr, nullptr, 0, 0);
    } else {
        // ─── LARGE GRID: interleaved bf16 staging → grouped dim1 quantize ───
        tk_silu_deriv_interleaved::launch(
            reinterpret_cast<const __nv_bfloat16*>(dh.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(h13.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dh13_bf16.data_ptr()),
            M, H, stream);

        (void)tk_group_quantize_dim1_launch(
            dh13_bf16, {H, H},
            fp4_row_full, fp4_col_full, sg_buf,
            amax_buf, sync_buf, psync_buf,
            tma_host_buf, tma_dev_buf,
            {out1_sc, out2_sc},
            {out1_fp4_t, out2_fp4_t},
            {out1_sc_t, out2_sc_t},
            true);

        split_fp4_row_groups_from_full(
            fp4_row_full, out1_fp4, out2_fp4, M, H, stream);
        out1_fp4_t = fp4_col_full.narrow(0, 0, H);
        out2_fp4_t = fp4_col_full.narrow(0, H, H);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tk_silu_deriv_quantize_for_gemm_launch failed: ", cudaGetErrorString(err));

    auto opts_f32 = torch::dtype(torch::kFloat32).device(dh.device());
    return std::make_tuple(
        out1_fp4.view(torch::kFloat4_e2m1fn_x2), out1_sc.view(torch::kFloat8_e4m3fn),
        out1_fp4_t.view(torch::kFloat4_e2m1fn_x2), out1_sc_t.view(torch::kFloat8_e4m3fn),
        sg_buf.narrow(0, 0, 1), torch::zeros({1}, opts_f32),
        out2_fp4.view(torch::kFloat4_e2m1fn_x2), out2_sc.view(torch::kFloat8_e4m3fn),
        out2_fp4_t.view(torch::kFloat4_e2m1fn_x2), out2_sc_t.view(torch::kFloat8_e4m3fn),
        sg_buf.narrow(0, 1, 1), torch::zeros({1}, opts_f32)
    );
}


// ═══════════════════════════════════════════════════════════════════
void tk_localcta_h_tile_quantize_out(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor r_tile,
    torch::Tensor row_outer_scales,
    torch::Tensor col_outer_scales,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    torch::Tensor work_counter) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
            input.scalar_type() == torch::kBFloat16,
        "localCTA H input must be contiguous CUDA bf16 [M,N]");
    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    TORCH_CHECK(
        rows % 256 == 0 && cols % 256 == 0,
        "localCTA H dimensions must be multiples of 256");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() && gamma.dim() == 1 &&
            gamma.scalar_type() == torch::kBFloat16 && gamma.numel() == cols,
        "localCTA H gamma must be contiguous CUDA bf16 [N]");
    TORCH_CHECK(
        r_tile.is_cuda() && r_tile.is_contiguous() &&
            r_tile.scalar_type() == torch::kFloat32 &&
            r_tile.sizes() == torch::IntArrayRef({rows / 128, cols / 128}),
        "localCTA H r_tile must be contiguous CUDA float32 [M/128,N/128]");
    TORCH_CHECK(
        row_outer_scales.is_cuda() && row_outer_scales.is_contiguous() &&
            row_outer_scales.scalar_type() == torch::kFloat32 &&
            row_outer_scales.numel() == rows / 256,
        "localCTA H row outer scales must contain M/256 float32 values");
    TORCH_CHECK(
        col_outer_scales.is_cuda() && col_outer_scales.is_contiguous() &&
            col_outer_scales.scalar_type() == torch::kFloat32 &&
            col_outer_scales.numel() == cols / 256,
        "localCTA H col outer scales must contain N/256 float32 values");
    TORCH_CHECK(
        row_fp4.is_cuda() && row_fp4.is_contiguous() &&
            row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2 &&
            row_fp4.sizes() == torch::IntArrayRef({rows, cols / 2}),
        "localCTA H row_fp4 shape/dtype mismatch");
    TORCH_CHECK(
        row_sc.is_cuda() && row_sc.is_contiguous() &&
            row_sc.scalar_type() == torch::kFloat8_e4m3fn &&
            row_sc.sizes() ==
                torch::IntArrayRef({rows / 128, cols / 64, 512}),
        "localCTA H row scales shape/dtype mismatch");
    TORCH_CHECK(
        col_fp4.is_cuda() && col_fp4.is_contiguous() &&
            col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2 &&
            col_fp4.sizes() == torch::IntArrayRef({cols, rows / 2}),
        "localCTA H col_fp4 shape/dtype mismatch");
    TORCH_CHECK(
        col_sc.is_cuda() && col_sc.is_contiguous() &&
            col_sc.scalar_type() == torch::kFloat8_e4m3fn &&
            col_sc.sizes() ==
                torch::IntArrayRef({cols / 128, rows / 64, 512}),
        "localCTA H col scales shape/dtype mismatch");
    TORCH_CHECK(
        work_counter.is_cuda() && work_counter.is_contiguous() &&
            work_counter.scalar_type() == torch::kInt32 &&
            work_counter.numel() == 1,
        "localCTA H work counter must be contiguous CUDA int32 [1]");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const int device_index = input.get_device();
    const torch::Tensor tensors[] = {
        gamma, r_tile, row_outer_scales, col_outer_scales,
        row_fp4, row_sc, col_fp4, col_sc, work_counter,
    };
    for (const auto& tensor : tensors) {
        TORCH_CHECK(
            tensor.get_device() == device_index,
            "localCTA H tensors must share one CUDA device");
    }
    auto stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    v5_check_cuda(
        cudaMemsetAsync(
            work_counter.data_ptr<int32_t>(), 0, sizeof(int32_t), stream),
        "resetting localCTA H work counter");

    alignas(64) CUtensorMap input_map{};
    alignas(64) CUtensorMap row_map{};
    alignas(64) CUtensorMap col_map{};
    alignas(64) CUtensorMap row_sc_map{};
    alignas(64) CUtensorMap col_sc_map{};
    create_tma_2d(
        input_map, input.data_ptr(), rows, cols,
        tk_v3::V3_BUFF_DIM_Y, tk_v3::V3_BUFF_DIM_X, cols, 16,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(
        row_map, row_fp4.data_ptr(), rows, cols,
        tk_v3::V3_BUFF_DIM_Y, tk_v3::V3_BUFF_DIM_X, cols, 4);
    create_tma_2d(
        col_map, col_fp4.data_ptr(), cols, rows,
        tk_v3::V3_BUFF_DIM_X, tk_v3::V3_BUFF_DIM_Y, rows, 4);
    const int64_t row_sc_x = (cols / 64) * 256;
    const int64_t col_sc_x = (rows / 64) * 256;
    create_tma_2d(
        row_sc_map, row_sc.data_ptr(), rows / 128, row_sc_x,
        1, 256, row_sc_x, 16);
    create_tma_2d(
        col_sc_map, col_sc.data_ptr(), cols / 128, col_sc_x,
        1, 256, col_sc_x, 16);

    const int smem = tk_v5_localcta_h_tile::dynamic_smem_size();
    auto kernel = tk_v5_localcta_h_tile::persistent_quantize_kernel;
    v5_check_cuda(
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
        "setting localCTA H dynamic shared memory");
    int blocks_per_sm = 0;
    v5_check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, kernel, tk_v3::V3_THREADS, smem),
        "querying localCTA H occupancy");
    int num_sms = 0;
    v5_check_cuda(
        cudaDeviceGetAttribute(
            &num_sms, cudaDevAttrMultiProcessorCount, device_index),
        "querying localCTA H SM count");
    const int total_tiles = static_cast<int>((rows / 128) * (cols / 128));
    const int blocks = std::min(total_tiles, blocks_per_sm * num_sms);
    TORCH_CHECK(blocks > 0, "localCTA H kernel has zero occupancy");

    kernel<<<blocks, tk_v3::V3_THREADS, smem, stream>>>(
        input_map, row_map, col_map, row_sc_map, col_sc_map,
        r_tile.data_ptr<float>(),
        reinterpret_cast<const IType*>(gamma.data_ptr()),
        row_outer_scales.data_ptr<float>(),
        col_outer_scales.data_ptr<float>(),
        rows, cols,
        reinterpret_cast<unsigned int*>(work_counter.data_ptr<int32_t>()),
        total_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tk_h_tile_backward_out(
    torch::Tensor du,
    torch::Tensor z,
    torch::Tensor gamma,
    torch::Tensor r_tile,
    torch::Tensor dx,
    torch::Tensor dgamma_partial,
    torch::Tensor dgamma) {
    TORCH_CHECK(
        du.is_cuda() && du.is_contiguous() && du.dim() == 2 &&
            du.scalar_type() == torch::kBFloat16,
        "H backward du must be contiguous CUDA bf16 [M,N]");
    const int64_t rows = du.size(0);
    const int64_t cols = du.size(1);
    TORCH_CHECK(
        rows > 0 && cols > 0 && rows % 128 == 0 && cols % 128 == 0,
        "H backward dimensions must be positive multiples of 128");
    TORCH_CHECK(
        z.is_cuda() && z.is_contiguous() && z.scalar_type() == torch::kBFloat16 &&
            z.sizes() == du.sizes(),
        "H backward z must match du");
    TORCH_CHECK(
        gamma.is_cuda() && gamma.is_contiguous() && gamma.dim() == 1 &&
            gamma.scalar_type() == torch::kBFloat16 && gamma.numel() == cols,
        "H backward gamma must be contiguous CUDA bf16 [N]");
    TORCH_CHECK(
        r_tile.is_cuda() && r_tile.is_contiguous() &&
            r_tile.scalar_type() == torch::kFloat32 &&
            r_tile.sizes() == torch::IntArrayRef({rows / 128, cols / 128}),
        "H backward r_tile shape/dtype mismatch");
    TORCH_CHECK(
        dx.is_cuda() && dx.is_contiguous() && dx.scalar_type() == torch::kBFloat16 &&
            dx.sizes() == du.sizes(),
        "H backward dx must match du");
    TORCH_CHECK(
        dgamma_partial.is_cuda() && dgamma_partial.is_contiguous() &&
            dgamma_partial.scalar_type() == torch::kFloat32 &&
            dgamma_partial.sizes() == torch::IntArrayRef({rows / 128, cols}),
        "H backward dgamma_partial must be float32 [M/128,N]");
    TORCH_CHECK(
        dgamma.is_cuda() && dgamma.is_contiguous() && dgamma.dim() == 1 &&
            dgamma.scalar_type() == torch::kBFloat16 && dgamma.numel() == cols,
        "H backward dgamma must be CUDA bf16 [N]");

    const c10::cuda::CUDAGuard device_guard(du.device());
    const int device_index = du.get_device();
    const torch::Tensor tensors[] = {
        z, gamma, r_tile, dx, dgamma_partial, dgamma,
    };
    for (const auto& tensor : tensors) {
        TORCH_CHECK(
            tensor.get_device() == device_index,
            "H backward tensors must share one CUDA device");
    }
    auto stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    tk_h_tile_backward::backward_kernel<<<
        dim3(cols / 128, rows / 128), tk_h_tile_backward::THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(du.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(z.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.data_ptr()),
        r_tile.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr()),
        dgamma_partial.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols));
    tk_h_tile_backward::dgamma_reduce_kernel<<<
        (cols + 255) / 256, 256, 0, stream>>>(
        dgamma_partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dgamma.data_ptr()),
        static_cast<int>(rows / 128),
        static_cast<int>(cols));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Pybind11 bindings
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Split3CaptureState, std::shared_ptr<Split3CaptureState>>(
        m, "Split3CaptureState")
        .def_property_readonly(
            "tensor_manifest", &Split3CaptureState::tensor_manifest,
            "Copied, read-only metadata for every descriptor/input/output tensor")
        .def_property_readonly(
            "caller_stream", &Split3CaptureState::stream_manifest,
            "CUDA stream pointer frozen when descriptor state was built");
    m.def("tk_quantize_for_gemm", &tk_v4_quantize_for_gemm,
          "v5 hybrid: quantize for GEMM (TMA scale output)",
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_quantize_weight_2d", &tk_quantize_weight_2d,
          "Native shared-16x16 v5 weight quantization for consistent forward and dgrad");
    m.def(
        "tk_gated_group_rmsnorm_quantize_for_gemm",
        &tk_gated_group_rmsnorm_quantize_for_gemm,
        "Native Nemotron gated group-RMSNorm to regular v5 FP4 producer",
        py::arg("scan"),
        py::arg("gate"),
        py::arg("gamma"),
        py::arg("epsilon"),
        py::arg("encode_centric"));
    m.def("tk_quantize_for_gemm_padded", &tk_v4_quantize_for_gemm_padded,
          "v5 hybrid: quantize a logical input directly into a zero-padded GEMM extent",
          py::arg("input"), py::arg("output_rows"), py::arg("output_cols"),
          py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_quantize_for_gemm_delayed", &tk_v4_quantize_for_gemm_delayed,
          "v5 hybrid: quantize for GEMM using previous amax (TMA scale output)",
          py::arg("input"), py::arg("prev_amax"), py::arg("return_transpose"),
          py::arg("encode_centric") = true, py::arg("collect_current_amax") = true);
    m.def("tk_quantize_nhsd_wo_for_gemm", &tk_quantize_nhsd_wo_for_gemm,
          "v5 hybrid: quantize logical WO matrix directly from contiguous [B,H,S,D]",
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric") = true);
    m.def("tk_quantize_for_gemm_constant_scale", &tk_quantize_for_gemm_constant_scale,
          "Unit-bound quantize: Phase-2 only with S_enc=2688 and sg=1/2688 (skips amax scan)",
          py::arg("input"), py::arg("return_transpose"));
    m.def("tk_quantize_for_gemm_opt", &tk_quantize_for_gemm_opt,
          "Regular-TK opt quantize with data SR and block-16 RHT support",
          py::arg("input"), py::arg("return_transpose") = true, py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0,
          py::arg("global_scale_target") = kDefaultNvfp4GlobalScaleTarget,
          py::arg("data_sr_axes") = "both");
    m.def("tk_sqrelu_quantize_for_gemm_opt", &tk_sqrelu_quantize_for_gemm_opt,
          "Regular-TK fused square-ReLU activation + opt quantize",
          py::arg("input"), py::arg("return_transpose") = true, py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0);
    m.def("tk_sqrelu_deriv_quantize_for_gemm_opt", &tk_sqrelu_deriv_quantize_for_gemm_opt,
          "Regular-TK fused square-ReLU derivative + opt quantize",
          py::arg("dh"), py::arg("h1_raw"), py::arg("return_transpose") = true, py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "none",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0);
    m.def("tk_quantize_col_only", &tk_quantize_col_only,
          "Col-only quantize: takes pre-computed sg, produces only col FP4 + scales",
          py::arg("input"), py::arg("sg"));
    m.def("tk_quantize_mxfp4_row_nvfp4_col",
          &tk_quantize_mxfp4_row_nvfp4_col,
          "MXFP4 row plus native-v5 NVFP4 column with a shared amax pass",
          py::arg("input"), py::arg("threads") = 256,
          py::arg("data_stochastic_rounding") = false,
          py::arg("rng_seed") = 0ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("tk_quantize_row_for_gemm_sr", &tk_quantize_row_for_gemm_sr,
          "Row-only GEMM quantize with stochastic rounding, returns (row_fp4,row_sc,sg,sg)",
          py::arg("input"), py::arg("encode_centric") = true);
    m.def("tk_quantize_transpose", &tk_quantize_transpose,
          "quantize with pre-computed amax",
          py::arg("input"), py::arg("amax_row"), py::arg("amax_col"), py::arg("return_transpose"),
          py::arg("stochastic_rounding") = false,
          py::arg("encode_centric") = true,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("row_rht") = false,
          py::arg("col_rht") = false,
          py::arg("return_row") = true,
          py::arg("rng_state") = torch::Tensor());
    m.def("tk_group_quantize_for_gemm", &tk_group_quantize_for_gemm,
          "Grouped NVFP4 quantise — per-split amax (dim=0)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_for_gemm_v2", &tk_group_quantize_for_gemm_v2,
          "Grouped NVFP4 quantise — non-persistent two-pass (multi-stream safe)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_split_for_gemm_v2", &tk_group_quantize_split_for_gemm_v2,
          "Split-input FFN weight quantise — non-persistent two-pass grouped layout",
          py::arg("w1"), py::arg("w3"));
    m.def("tk_group_quantize_v2_alloc", &tk_group_quantize_v2_alloc,
          "Pre-allocate buffers for v2 grouped quant (call on capture stream)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_v2_launch", &tk_group_quantize_v2_launch,
          "Kernel-only v2 grouped quant using pre-allocated buffers (graph-safe on any stream)",
          py::arg("input"), py::arg("split_sections"),
          py::arg("wc_fp4_row"), py::arg("amaxes"), py::arg("sg_cat"),
          py::arg("fwd_b_sg"), py::arg("dgrad_b_sg"), py::arg("d_offsets"),
          py::arg("sc_row_list"), py::arg("fp4_col_list"), py::arg("sc_col_list"));
    m.def("tk_group_quantize_v5_alloc", &tk_group_quantize_v5_alloc,
          "Pre-allocate buffers for v5 persistent/fused grouped quant (call on capture stream)",
          py::arg("input"), py::arg("split_sections"));
    m.def("tk_group_quantize_v5_launch", &tk_group_quantize_v5_launch,
          "Kernel-only v5 persistent/fused grouped quant using pre-allocated buffers (graph-safe on any stream)",
          py::arg("input"), py::arg("split_sections"),
          py::arg("wc_fp4_row"), py::arg("wc_fp4_col"), py::arg("sg_cat"),
          py::arg("fwd_b_sg"), py::arg("dgrad_b_sg"),
          py::arg("amax_tensor"), py::arg("sync_tensor"), py::arg("psync_tensor"),
          py::arg("tma_dev_buf"),
          py::arg("sc_row_list"), py::arg("fp4_col_list"), py::arg("sc_col_list"));
    m.def("tk_group_quantize_dim1_for_gemm", &tk_group_quantize_dim1_for_gemm,
          "Grouped NVFP4 quantise — per-column-group amax (dim=1). Returns: "
          "(fp4_row_list, sc_row_list, sg, fp4_col_list, sc_col_list, fp4_row_full, sc_row_cat, fp4_col_full, sc_col_cat)",
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_group_quantize_dim1_split3_for_gemm", &tk_group_quantize_dim1_split3_for_gemm,
          "Grouped NVFP4 dim1 split3 quantise without materializing the BF16 concatenation",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0,
          py::arg("data_sr_axes") = "both");
    m.def("tk_group_quantize_dim1_split3_capture_alloc",
          &tk_group_quantize_dim1_split3_capture_alloc,
          "Build exact-pointer split3 graph state from an eager split3 result",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("fp4_row_full"), py::arg("fp4_col_full"),
          py::arg("sg_per_group"), py::arg("sc_row_allocs"),
          py::arg("fp4_col_allocs"), py::arg("sc_col_allocs"),
          py::arg("sc_row_cat"), py::arg("sc_col_cat"),
          py::arg("tma_dev_buf"), py::arg("tma_host_buf"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0,
          py::arg("data_sr_axes") = "both");
    m.def("tk_group_quantize_dim1_split3_launch",
          &tk_group_quantize_dim1_split3_launch,
          "No-CUDA-allocation exact-pointer split3 launch for CUDA graph capture",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("state"));
    m.def("tk_group_quantize_dim1_alloc", &tk_group_quantize_dim1_alloc,
          "Pre-allocate buffers for dim1 grouped quant (call BEFORE graph capture)",
          py::arg("input"), py::arg("col_split_sections"));
    m.def("tk_group_quantize_dim1_launch", &tk_group_quantize_dim1_launch,
          "Kernel-only dim1 grouped quant using pre-allocated buffers (graph-safe inside capture)",
          py::arg("input"), py::arg("col_split_sections"),
          py::arg("fp4_row_full"), py::arg("fp4_col_full"), py::arg("sg_per_group"),
          py::arg("amax_tensor"), py::arg("sync_tensor"), py::arg("psync_tensor"),
          py::arg("tma_host_buf"), py::arg("tma_dev_buf"),
          py::arg("sc_row_allocs"), py::arg("fp4_col_allocs"), py::arg("sc_col_allocs"),
          py::arg("skip_cat") = false);
    m.def("tk_fused_norm_quantize", &tk_fused_norm_quantize,
          "Fused RMSNorm + optional SiLU + FP4 quantize (single GMEM pass)",
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("with_silu"), py::arg("return_transpose"));
    m.def("tk_fused_norm_quantize_from_inv_rms",
          &tk_fused_norm_quantize_from_inv_rms,
          "RMSNorm + FP4 quantize using caller-provided inverse RMS",
          py::arg("input"), py::arg("gamma"), py::arg("inv_rms"),
          py::arg("with_silu"), py::arg("return_transpose"));
    m.def("tk_fused_norm_quantize_from_row_rms_partial",
          &tk_fused_norm_quantize_from_row_rms_partial,
          "Reduce GEMM row RMS partials then RMSNorm + FP4 quantize",
          py::arg("input"), py::arg("gamma"), py::arg("row_rms_partial"),
          py::arg("epsilon"), py::arg("with_silu"),
          py::arg("return_transpose"));
    m.def("tk_localcta_h_tile_quantize_out", &tk_localcta_h_tile_quantize_out,
          "Phase-2-only persistent tile-RMS localCTA NVFP4 quantization",
          py::arg("input"), py::arg("gamma"), py::arg("r_tile"),
          py::arg("row_outer_scales"), py::arg("col_outer_scales"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("work_counter"));
    m.def("tk_h_tile_backward_out", &tk_h_tile_backward_out,
          "Native tile-RMS backward with deterministic dgamma reduction",
          py::arg("du"), py::arg("z"), py::arg("gamma"), py::arg("r_tile"),
          py::arg("dx"), py::arg("dgamma_partial"), py::arg("dgamma"));
    m.def("tk_fused_norm_quantize_with_output", &tk_fused_norm_quantize_with_output,
          "Fused RMSNorm + BF16 output + optional SiLU + FP4 quantize",
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("with_silu"), py::arg("return_transpose"));
    m.def("tk_fused_norm_quantize_opt", &tk_fused_norm_quantize_opt,
          "Fused RMSNorm + regular-TK row-RHT/SR-aware FP4 quantize",
          py::arg("input"), py::arg("gamma"), py::arg("epsilon"),
          py::arg("return_transpose") = true,
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "row",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0);
    m.def("tk_silu_quantize_for_gemm", &tk_silu_quantize_for_gemm,
          "Fused silu(h1)*h3 + FP4 quantize from (M,2H) buffer",
          py::arg("h13"), py::arg("H"));
    m.def("tk_silu_mul_split_bf16", &tk_silu_mul_split_bf16,
          "Split-input silu(h1_raw)*h3 -> bf16 helper",
          py::arg("h1_raw"), py::arg("h3"));
    m.def("tk_silu_quantize_split_for_gemm", &tk_silu_quantize_split_for_gemm,
          "Fused silu(h1_raw)*h3 + FP4 quantize from split (M,H) tensors",
          py::arg("h1_raw"), py::arg("h3"));
    m.def("tk_silu_deriv_quantize_for_gemm", &tk_silu_deriv_quantize_for_gemm,
          "Fused SiLU derivative + dual FP4 quantize (backward)",
          py::arg("dh"), py::arg("h13"), py::arg("H"),
          py::arg("use_delayed_scaling") = false);
    m.def("tk_silu_deriv_quantize_for_gemm_delayed", &tk_silu_deriv_quantize_for_gemm_delayed,
          "Fused SiLU derivative + dual FP4 quantize using previous-iteration amax",
          py::arg("dh"), py::arg("h13"), py::arg("H"), py::arg("prev_amax"),
          py::arg("collect_current_amax") = true);
    m.def("tk_silu_deriv_split_bf16", &tk_silu_deriv_split_bf16,
          "Split-input silu-derivative helper -> 2x bf16",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_silu_deriv_quantize_split_for_gemm", &tk_silu_deriv_quantize_split_for_gemm,
          "Fused SiLU derivative + dual FP4 quantize from split (M,H) tensors",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));
    m.def("tk_silu_deriv_quantize_split_for_gemm_delayed", &tk_silu_deriv_quantize_split_for_gemm_delayed,
          "Fused SiLU derivative + dual FP4 quantize from split tensors using previous amax",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("prev_amax"), py::arg("collect_current_amax") = true);
    m.def("tk_silu_deriv_quantize_split_for_gemm_opt", &tk_silu_deriv_quantize_split_for_gemm_opt,
          "Split SiLU derivative + row-RHT/SR-aware regular-TK FP4 quantize",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("return_transpose") = true,
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "row",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0,
          py::arg("data_sr_axes") = "both");
    m.def("tk_silu_deriv_quantize_split_for_gemm_opt_alloc", &tk_silu_deriv_quantize_split_for_gemm_opt_alloc,
          "Pre-allocate buffers for row-RHT/SR-aware split silu_deriv+regular-TK quantize",
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_silu_deriv_quantize_split_for_gemm_opt_launch", &tk_silu_deriv_quantize_split_for_gemm_opt_launch,
          "Kernel-only row-RHT/SR-aware split silu_deriv+regular-TK quantize using pre-allocated buffers",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("return_transpose") = true,
          py::arg("encode_centric") = true,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_axes") = "row",
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 0,
          py::arg("rng_subsequence") = 0,
          py::arg("out1_fp4"), py::arg("out1_sc"),
          py::arg("out1_fp4_t"), py::arg("out1_sc_t"),
          py::arg("out2_fp4"), py::arg("out2_sc"),
          py::arg("out2_fp4_t"), py::arg("out2_sc_t"),
          py::arg("sgs"), py::arg("amaxes"),
          py::arg("sync_buf"),
          py::arg("data_sr_axes") = "both");
    m.def("tk_silu_deriv_quantize_split_for_gemm_alloc", &tk_silu_deriv_quantize_split_for_gemm_alloc,
          "Pre-allocate buffers for split silu_deriv+quantize",
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_silu_deriv_quantize_split_for_gemm_launch", &tk_silu_deriv_quantize_split_for_gemm_launch,
          "Kernel-only split silu_deriv+quantize using pre-allocated buffers",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("out1_fp4"), py::arg("out1_sc"),
          py::arg("out1_fp4_t"), py::arg("out1_sc_t"),
          py::arg("out2_fp4"), py::arg("out2_sc"),
          py::arg("out2_fp4_t"), py::arg("out2_sc_t"),
          py::arg("sg_buf"), py::arg("zero_buf"),
          py::arg("amax_buf"), py::arg("sync_buf"),
          py::arg("dh1"), py::arg("dh3"));
    m.def("bf16_transpose_into", &bf16_transpose_into,
          "Tiled BF16 transpose into a preallocated contiguous output",
          py::arg("src"), py::arg("dst"));

    // ── CUDA graph-safe alloc/launch split APIs ──
    m.def("tk_v4_quantize_for_gemm_alloc", &tk_v4_quantize_for_gemm_alloc,
          "Pre-allocate buffers for v4 quantize (call BEFORE graph capture)",
          py::arg("M"), py::arg("K"), py::arg("return_transpose"), py::arg("device"));
    m.def("tk_v4_quantize_for_gemm_launch", &tk_v4_quantize_for_gemm_launch,
          "Kernel-only v4 quantize using pre-allocated buffers (graph-safe)",
          py::arg("input"), py::arg("return_transpose"), py::arg("encode_centric") = true,
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("amax_buf"), py::arg("sync_buf"));
    m.def("tk_silu_deriv_quantize_for_gemm_alloc", &tk_silu_deriv_quantize_for_gemm_alloc,
          "Pre-allocate buffers for silu_deriv+quantize (call BEFORE graph capture)",
          py::arg("M"), py::arg("H"), py::arg("device"));
    m.def("tk_silu_deriv_quantize_for_gemm_launch", &tk_silu_deriv_quantize_for_gemm_launch,
          "Kernel-only silu_deriv+quantize using pre-allocated buffers (graph-safe)",
          py::arg("dh"), py::arg("h13"), py::arg("H"),
          py::arg("out1_fp4"), py::arg("out1_sc"),
          py::arg("out1_fp4_t"), py::arg("out1_sc_t"),
          py::arg("out2_fp4"), py::arg("out2_sc"),
          py::arg("out2_fp4_t"), py::arg("out2_sc_t"),
          py::arg("sg_buf"), py::arg("amax_buf"),
          py::arg("sync_buf"), py::arg("psync_buf"),
          py::arg("fp4_row_full"), py::arg("fp4_col_full"),
          py::arg("dh13_bf16"),
          py::arg("tma_host_buf"), py::arg("tma_dev_buf"));

    // ── Batched quantize: N tensors in single C++ call ──
    m.def("tk_batched_quantize_for_gemm", [](
        const std::vector<torch::Tensor> &inputs,
        bool return_transpose,
        bool encode_centric
    ) -> std::tuple<
        std::vector<torch::Tensor>, std::vector<torch::Tensor>,
        std::vector<torch::Tensor>, std::vector<torch::Tensor>,
        std::vector<torch::Tensor>, std::vector<torch::Tensor>,
        std::vector<torch::Tensor>, std::vector<torch::Tensor>
    > {
        const int n = (int)inputs.size();
        std::vector<torch::Tensor> row_fp4s, row_scs, col_fp4s, col_scs, sgs, sgs2;
        std::vector<torch::Tensor> amax_keepalive, sync_keepalive;
        row_fp4s.reserve(n); row_scs.reserve(n);
        col_fp4s.reserve(n); col_scs.reserve(n);
        sgs.reserve(n); sgs2.reserve(n);
        amax_keepalive.reserve(n); sync_keepalive.reserve(n);
        for (int i = 0; i < n; ++i) {
            auto [rf, rs, cf, cs, sg, sg2, amax_buf, sync_buf] = tk_v4_quantize_for_gemm(
                inputs[i], return_transpose, encode_centric);
            row_fp4s.push_back(std::move(rf));
            row_scs.push_back(std::move(rs));
            col_fp4s.push_back(std::move(cf));
            col_scs.push_back(std::move(cs));
            sgs.push_back(std::move(sg));
            sgs2.push_back(std::move(sg2));
            amax_keepalive.push_back(std::move(amax_buf));
            sync_keepalive.push_back(std::move(sync_buf));
        }
        return {
            row_fp4s, row_scs, col_fp4s, col_scs,
            sgs, sgs2, amax_keepalive, sync_keepalive,
        };
    },
    "Batched quantize: N tensors in single C++ call (eliminates N-1 Python boundary crossings)",
    py::arg("inputs"), py::arg("return_transpose"), py::arg("encode_centric") = true);
}
