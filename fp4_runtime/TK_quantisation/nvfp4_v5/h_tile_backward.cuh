#pragma once

#include <cuda_bf16.h>

namespace tk_h_tile_backward {

constexpr int TILE = 128;
constexpr int THREADS = 256;
constexpr float TILE_INV_ELEMENTS = 0x1p-14f;

__global__ void backward_kernel(
    const __nv_bfloat16* __restrict__ du,
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ r_tile,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dgamma_partial,
    int rows,
    int cols) {
    const int tile_col = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int tid = threadIdx.x;
    const int row0 = tile_row * TILE;
    const int col0 = tile_col * TILE;
    __shared__ float dot_s[THREADS];

    float dot = 0.0f;
    #pragma unroll
    for (int e = tid; e < TILE * TILE; e += THREADS) {
        const int row = row0 + e / TILE;
        const int col = col0 + e % TILE;
        const int offset = row * cols + col;
        const float zv = __bfloat162float(z[offset]);
        const float gv =
            __bfloat162float(du[offset]) * __bfloat162float(gamma[col]);
        dot = fmaf(gv, zv, dot);
    }
    dot_s[tid] = dot;
    __syncthreads();
    #pragma unroll
    for (int stride = THREADS / 2; stride; stride >>= 1) {
        if (tid < stride) {
            dot_s[tid] += dot_s[tid + stride];
        }
        __syncthreads();
    }

    const float r = r_tile[tile_row * (cols / TILE) + tile_col];
    const float correction = r * r * r * dot_s[0] * TILE_INV_ELEMENTS;
    #pragma unroll
    for (int e = tid; e < TILE * TILE; e += THREADS) {
        const int row = row0 + e / TILE;
        const int col = col0 + e % TILE;
        const int offset = row * cols + col;
        const float zv = __bfloat162float(z[offset]);
        const float gv =
            __bfloat162float(du[offset]) * __bfloat162float(gamma[col]);
        dx[offset] = __float2bfloat16_rn(fmaf(-zv, correction, r * gv));
    }

    if (tid < TILE) {
        const int col = col0 + tid;
        float dgamma = 0.0f;
        #pragma unroll
        for (int row = 0; row < TILE; ++row) {
            const int offset = (row0 + row) * cols + col;
            const float normalized = __bfloat162float(z[offset]) * r;
            dgamma = fmaf(__bfloat162float(du[offset]), normalized, dgamma);
        }
        dgamma_partial[tile_row * cols + col] = dgamma;
    }
}

__global__ void dgamma_reduce_kernel(
    const float* __restrict__ partial,
    __nv_bfloat16* __restrict__ dgamma,
    int tile_rows,
    int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    float sum = 0.0f;
    for (int tile_row = 0; tile_row < tile_rows; ++tile_row) {
        sum += partial[tile_row * cols + col];
    }
    dgamma[col] = __float2bfloat16_rn(sum);
}

}  // namespace tk_h_tile_backward
