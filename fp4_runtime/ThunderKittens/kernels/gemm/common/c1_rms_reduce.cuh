#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>

namespace c1_rms_reduce {

__global__ void row_rms_coeff_warp_kernel(
    const float* __restrict__ row_rms_partial,
    float* __restrict__ coeff,
    int64_t rows,
    int64_t partial_cols,
    int64_t hidden_size,
    float eps
) {
    constexpr int WARPS_PER_BLOCK = 4;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    if (row >= rows) {
        return;
    }

    const int64_t base = row * partial_cols;
    float sum = 0.0f;
    for (int64_t col = lane; col < partial_cols; col += 32) {
        sum += row_rms_partial[base + col];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        coeff[row] = rsqrtf(sum / static_cast<float>(hidden_size) + eps);
    }
}

inline void row_rms_reduce_entrypoint(
    const at::Tensor& row_rms_partial,
    at::Tensor& coeff,
    int64_t hidden_size,
    double eps
) {
    TORCH_CHECK(row_rms_partial.is_cuda() && row_rms_partial.is_contiguous(),
                "row_rms_partial must be contiguous CUDA tensor");
    TORCH_CHECK(row_rms_partial.scalar_type() == at::kFloat &&
                    row_rms_partial.dim() == 2,
                "row_rms_partial must be float32 [M,P]");
    TORCH_CHECK(coeff.is_cuda() && coeff.is_contiguous() &&
                    coeff.scalar_type() == at::kFloat && coeff.dim() == 1,
                "coeff must be contiguous CUDA float32 [M]");
    TORCH_CHECK(row_rms_partial.size(0) == coeff.size(0),
                "coeff length must match row_rms_partial rows");
    TORCH_CHECK(hidden_size > 0 && hidden_size % 32 == 0,
                "hidden_size must be a positive multiple of 32");
    TORCH_CHECK(row_rms_partial.size(1) > 0 &&
                    row_rms_partial.size(1) <= hidden_size / 32 &&
                    hidden_size % row_rms_partial.size(1) == 0,
                "row_rms_partial columns must evenly partition hidden_size");
    TORCH_CHECK(row_rms_partial.size(1) <= 128,
                "warp reducer supports at most 128 partial columns");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0,
                "eps must be finite and non-negative");
    TORCH_CHECK(row_rms_partial.get_device() == coeff.get_device(),
                "row_rms_partial and coeff must be on the same device");
    at::assert_no_overlap(coeff, row_rms_partial);

    const int64_t rows = row_rms_partial.size(0);
    if (rows == 0) {
        return;
    }
    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 block(32 * WARPS_PER_BLOCK);
    row_rms_coeff_warp_kernel<<<
        grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        row_rms_partial.data_ptr<float>(),
        coeff.data_ptr<float>(),
        rows,
        row_rms_partial.size(1),
        hidden_size,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace c1_rms_reduce
