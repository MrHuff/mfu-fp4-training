// Nemotron-H gated group-RMSNorm + exact amax for regular NVFP4 v5.
//
// This pass writes the BF16-rounded normalized tensor while atomically
// reducing its exact tensor-global amax. The existing v5 phase-2-only
// producer then reads that tensor once to emit row/column FP4 and scales.

#pragma once

#include <cuda_bf16.h>

#include "persistent_quantize.cuh"

namespace gated_group_rmsnorm_quant {

using namespace transformer_engine;
using namespace tk_v3;

#if FP4_TYPE_SUPPORTED

static constexpr int kHidden = 8192;
static constexpr int kGroupSize = 1024;
static constexpr int kGroupsPerRow = kHidden / kGroupSize;
static constexpr int kThreads = 256;
static constexpr int kValuesPerThread = kGroupSize / kThreads;

__device__ __forceinline__ float gated_silu(float value) {
    // gated_rmsnorm_fwd is built with --use_fast_math.
    return __fdividef(value, 1.0f + __expf(-value));
}

__device__ __forceinline__ float block_sum_256(
    float value,
    float* warp_values
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    float result = threadIdx.x < 8 ? warp_values[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            result += __shfl_down_sync(
                0xffffffff, result, offset);
        }
        if (lane == 0) {
            warp_values[0] = result;
        }
    }
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ float block_max_256(
    float value,
    float* warp_values
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffff, value, offset));
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();

    float result = threadIdx.x < 8 ? warp_values[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            result = fmaxf(
                result,
                __shfl_down_sync(0xffffffff, result, offset));
        }
        if (lane == 0) {
            warp_values[0] = result;
        }
    }
    __syncthreads();
    return warp_values[0];
}

__global__ void __launch_bounds__(kThreads)
gated_group_rmsnorm_amax_kernel(
    const IType* __restrict__ scan,
    const IType* __restrict__ gate,
    const IType* __restrict__ gamma,
    IType* __restrict__ normalized,
    float* __restrict__ inv_rms,
    float* __restrict__ global_amax,
    int rows,
    int64_t gate_row_stride,
    float epsilon
) {
    __shared__ float warp_values[8];
    const int total_groups = rows * kGroupsPerRow;
    float cta_max = 0.0f;

    for (int group = blockIdx.x;
         group < total_groups;
         group += gridDim.x) {
        const int row = group / kGroupsPerRow;
        const int row_group = group % kGroupsPerRow;
        const int col_base = row_group * kGroupSize;
        const int64_t scan_base =
            static_cast<int64_t>(row) * kHidden + col_base;
        const int64_t gate_base =
            static_cast<int64_t>(row) * gate_row_stride + col_base;
        float values[kValuesPerThread];
        float sumsq = 0.0f;

#pragma unroll
        for (int item = 0; item < kValuesPerThread; ++item) {
            const int local = threadIdx.x + item * kThreads;
            const float scan_value =
                __bfloat162float(scan[scan_base + local]);
            const float gate_value =
                __bfloat162float(gate[gate_base + local]);
            values[item] = scan_value * gated_silu(gate_value);
            sumsq += values[item] * values[item];
        }

        const float inv = rsqrtf(
            block_sum_256(sumsq, warp_values) /
                static_cast<float>(kGroupSize) +
            epsilon);
        if (threadIdx.x == 0) {
            inv_rms[group] = inv;
        }

#pragma unroll
        for (int item = 0; item < kValuesPerThread; ++item) {
            const int local = threadIdx.x + item * kThreads;
            const IType rounded = __float2bfloat16_rn(
                values[item] * inv *
                __bfloat162float(gamma[col_base + local]));
            normalized[scan_base + local] = rounded;
            cta_max = fmaxf(
                cta_max,
                fabsf(__bfloat162float(rounded)));
        }
    }

    const float reduced_max = block_max_256(cta_max, warp_values);
    if (threadIdx.x == 0 && reduced_max > 0.0f) {
        transformer_engine::atomicMaxFloat(
            global_amax, reduced_max);
    }
}

#endif

}  // namespace gated_group_rmsnorm_quant
