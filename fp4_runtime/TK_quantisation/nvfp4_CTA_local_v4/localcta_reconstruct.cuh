#ifndef TK_LOCALCTA_RECONSTRUCT_CUH_
#define TK_LOCALCTA_RECONSTRUCT_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace tk_localcta_reconstruct {

enum SgGeometry : int {
    SG_CHUNK_GRID_128 = 0,
    SG_TILE_GRID_256 = 1,
    SG_OUTER_SCALE = 2,
};

__device__ __forceinline__ int scale_swizzle_idx(int row, int k_block, int k_blocks) {
    const int m_block = row / 128;
    const int k_block_groups = k_blocks / 4;
    const int k_block_group = k_block / 4;
    const int row_in_32 = row % 32;
    const int tile_in_block = (row / 32) % 4;
    const int kb_in_block = k_block % 4;

    const int block_base = (m_block * k_block_groups + k_block_group) * 512;
    const int local_idx = row_in_32 * 16 + tile_in_block * 4 + kb_in_block;
    return block_base + local_idx;
}

__device__ __forceinline__ int64_t sg_index(
    int row,
    int col,
    int sg_cols,
    int sg_geometry
) {
    if (sg_geometry == SG_OUTER_SCALE) {
        // Both row SG [ceil(rows / 256), 1] and transposed-column SG
        // [1, ceil(rows / 256)] are contiguous and flatten in row order.
        return row / 256;
    }
    const int tile_size =
        sg_geometry == SG_TILE_GRID_256 ? 256 : 128;
    const int tile_row = row / tile_size;
    const int tile_col = col / tile_size;
    return static_cast<int64_t>(tile_row) * sg_cols + tile_col;
}

__device__ __forceinline__ float unpack_fp4_elem(
    const __nv_fp4x2_e2m1 pair,
    int lane
) {
    const float2 vals = static_cast<float2>(pair);
    return lane == 0 ? vals.x : vals.y;
}

__global__ void reconstruct_rowwise_kernel(
    const __nv_fp4x2_e2m1* __restrict__ fp4,
    const __nv_fp8_e4m3* __restrict__ sc,
    const float* __restrict__ sg,
    __nv_bfloat16* __restrict__ out,
    int rows,
    int cols,
    int64_t sc_numel,
    int64_t sg_numel,
    int sg_cols,
    int sg_geometry
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t numel = (int64_t)rows * cols;
    if (idx >= numel) {
        return;
    }

    const int row = idx / cols;
    const int col = idx % cols;
    const int packed_idx = row * (cols / 2) + col / 2;
    const int k_block = col / 16;
    const int64_t scale_idx = scale_swizzle_idx(row, k_block, cols / 16);
    const int64_t global_scale_idx =
        sg_index(row, col, sg_cols, sg_geometry);
    // Host validation makes these branches unreachable for valid calls. Keep
    // the device-side guard as a final barrier against diagnostic OOB reads.
    if (scale_idx < 0 || scale_idx >= sc_numel ||
        global_scale_idx < 0 || global_scale_idx >= sg_numel) {
        out[idx] = __float2bfloat16(0.0f);
        return;
    }
    const float local_scale = static_cast<float>(sc[scale_idx]);
    const float global_scale = sg[global_scale_idx];
    const float x = unpack_fp4_elem(fp4[packed_idx], col & 1);
    out[idx] = __float2bfloat16(x * local_scale * global_scale);
}

}  // namespace tk_localcta_reconstruct

#endif  // TK_LOCALCTA_RECONSTRUCT_CUH_
