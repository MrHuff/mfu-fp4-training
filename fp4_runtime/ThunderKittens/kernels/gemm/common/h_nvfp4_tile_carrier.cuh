#pragma once

#include "kittens.cuh"
#include <cuda_bf16.h>

namespace h_nvfp4_tile_carrier {

constexpr int TILE = 128;
constexpr float TILE_INV_ELEMENTS = 0x1p-14f;
constexpr float LOCALCTA_SCALE_NUM = 1493.0f;

template <typename RT>
__device__ __forceinline__ float sumsq(const RT& x) {
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float a = __bfloat162float(x.tiles[i][j].data[k].x);
                const float b = __bfloat162float(x.tiles[i][j].data[k].y);
                s = fmaf(a, a, s);
                s = fmaf(b, b, s);
            }
        }
    }
    return s;
}

__device__ __forceinline__ float reduce_sum(
    float value,
    float* scratch) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, d);
    }
    if (kittens::warp::laneid() == 0) {
        scratch[kittens::warpgroup::warpid()] = value;
    }
    kittens::warpgroup::sync(1);
    if (kittens::warpgroup::warpid() == 0) {
        float total = kittens::warp::laneid() < kittens::WARPGROUP_WARPS
            ? scratch[kittens::warp::laneid()]
            : 0.0f;
        #pragma unroll
        for (int d = 16; d; d >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, d);
        }
        if (kittens::warp::laneid() == 0) {
            scratch[kittens::WARPGROUP_WARPS] = total;
        }
    }
    kittens::warpgroup::sync(1);
    return scratch[kittens::WARPGROUP_WARPS];
}

__device__ __forceinline__ float reduce_max(
    float value,
    float* scratch) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffu, value, d));
    }
    if (kittens::warp::laneid() == 0) {
        scratch[kittens::warpgroup::warpid()] = value;
    }
    kittens::warpgroup::sync(1);
    if (kittens::warpgroup::warpid() == 0) {
        float total = kittens::warp::laneid() < kittens::WARPGROUP_WARPS
            ? scratch[kittens::warp::laneid()]
            : 0.0f;
        #pragma unroll
        for (int d = 16; d; d >>= 1) {
            total = fmaxf(
                total,
                __shfl_down_sync(0xffffffffu, total, d));
        }
        if (kittens::warp::laneid() == 0) {
            scratch[kittens::WARPGROUP_WARPS] = total;
        }
    }
    kittens::warpgroup::sync(1);
    return scratch[kittens::WARPGROUP_WARPS];
}

__device__ __forceinline__ float tile_rsqrt(
    float sum,
    float* scratch,
    float eps) {
    const float total = reduce_sum(sum, scratch);
    if (kittens::warpgroup::warpid() == 0 &&
        kittens::warp::laneid() == 0) {
        scratch[kittens::WARPGROUP_WARPS] =
            rsqrtf(fmaf(total, TILE_INV_ELEMENTS, eps));
    }
    kittens::warpgroup::sync(1);
    return scratch[kittens::WARPGROUP_WARPS];
}

__device__ __forceinline__ float normalized_abs(
    kittens::bf16 value,
    float r,
    kittens::bf16 gamma) {
    const float scaled = (__bfloat162float(value) * r) *
        __bfloat162float(gamma);
    return fabsf(__bfloat162float(__float2bfloat16_rn(scaled)));
}

template <typename RT>
__device__ __forceinline__ float normalized_gamma_amax(
    const RT& z,
    float r,
    const kittens::bf16* gamma,
    int col_base) {
    const int lane = kittens::warp::laneid();
    const int lane_pair = lane & 3;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int c0 = col_base + j * 16 + lane_pair * 2;
            const int c1 = c0 + 8;
            const auto& q = z.tiles[i][j].data;
            amax = fmaxf(amax, normalized_abs(q[0].x, r, gamma[c0]));
            amax = fmaxf(amax, normalized_abs(q[0].y, r, gamma[c0 + 1]));
            amax = fmaxf(amax, normalized_abs(q[1].x, r, gamma[c0]));
            amax = fmaxf(amax, normalized_abs(q[1].y, r, gamma[c0 + 1]));
            amax = fmaxf(amax, normalized_abs(q[2].x, r, gamma[c1]));
            amax = fmaxf(amax, normalized_abs(q[2].y, r, gamma[c1 + 1]));
            amax = fmaxf(amax, normalized_abs(q[3].x, r, gamma[c1]));
            amax = fmaxf(amax, normalized_abs(q[3].y, r, gamma[c1 + 1]));
        }
    }
    return amax;
}

template <int BLOCK_SIZE = 256>
__global__ void localcta_reduce_kernel(
    const float* __restrict__ tile_amax,
    float* __restrict__ row_sg,
    float* __restrict__ col_sg,
    int tile_rows,
    int tile_cols) {
    kittens::pdl::wait();
    const int task = blockIdx.x;
    const int row_tasks = tile_rows / 2;
    float value = 0.0f;
    if (task < row_tasks) {
        const int tr0 = task * 2;
        for (int i = threadIdx.x; i < tile_cols * 2; i += BLOCK_SIZE) {
            const int tr = tr0 + i / tile_cols;
            const int tc = i % tile_cols;
            value = fmaxf(value, tile_amax[tr * tile_cols + tc]);
        }
    } else {
        const int col_task = task - row_tasks;
        const int tc0 = col_task * 2;
        for (int i = threadIdx.x; i < tile_rows * 2; i += BLOCK_SIZE) {
            const int tc = tc0 + i / tile_rows;
            const int tr = i % tile_rows;
            value = fmaxf(value, tile_amax[tr * tile_cols + tc]);
        }
    }
    __shared__ float scratch[BLOCK_SIZE];
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(
                scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (task < row_tasks) {
            row_sg[task] = scratch[0] / LOCALCTA_SCALE_NUM;
        } else {
            col_sg[task - row_tasks] = scratch[0] / LOCALCTA_SCALE_NUM;
        }
    }
}
}  // namespace h_nvfp4_tile_carrier
