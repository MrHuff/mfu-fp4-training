#pragma once

namespace tk_silu_deriv_interleaved {

constexpr int THREADS = 256;

__global__ void silu_deriv_dual_interleaved_kernel(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ h13,
    __nv_bfloat16* __restrict__ dh13,
    int64_t M,
    int64_t H
) {
    const int64_t total = M * H;
    const int64_t row_stride = 2 * H;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total / 2; i += stride) {
        const int64_t elem = i * 2;
        const int64_t row = elem / H;
        const int64_t col = elem % H;

        const __nv_bfloat162 dh_val =
            *reinterpret_cast<const __nv_bfloat162*>(dh + elem);
        const float2 dh_f = __bfloat1622float2(dh_val);

        const __nv_bfloat16* h1_ptr = h13 + row * row_stride + col;
        const __nv_bfloat16* h3_ptr = h13 + row * row_stride + H + col;
        const float2 h1_f =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h1_ptr));
        const float2 h3_f =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(h3_ptr));

        const float sigx = 1.0f / (1.0f + __expf(-h1_f.x));
        const float sigy = 1.0f / (1.0f + __expf(-h1_f.y));
        const float silux = h1_f.x * sigx;
        const float siluy = h1_f.y * sigy;
        const float silupx = sigx * (1.0f + h1_f.x - silux);
        const float silupy = sigy * (1.0f + h1_f.y - siluy);

        const float2 dh1_f = {
            dh_f.x * h3_f.x * silupx,
            dh_f.y * h3_f.y * silupy,
        };
        const float2 dh3_f = {
            dh_f.x * silux,
            dh_f.y * siluy,
        };

        __nv_bfloat16* dh1_out = dh13 + row * row_stride + col;
        __nv_bfloat16* dh3_out = dh13 + row * row_stride + H + col;
        *reinterpret_cast<__nv_bfloat162*>(dh1_out) = __float22bfloat162_rn(dh1_f);
        *reinterpret_cast<__nv_bfloat162*>(dh3_out) = __float22bfloat162_rn(dh3_f);
    }

    if ((total & 1) != 0 && idx == 0) {
        const int64_t i = total - 1;
        const int64_t row = i / H;
        const int64_t col = i % H;
        const float vd = __bfloat162float(dh[i]);
        const float v1 = __bfloat162float(h13[row * row_stride + col]);
        const float v3 = __bfloat162float(h13[row * row_stride + H + col]);
        const float sig = 1.0f / (1.0f + __expf(-v1));
        const float silu_v1 = v1 * sig;
        const float silup_v1 = sig * (1.0f + v1 - silu_v1);
        dh13[row * row_stride + col] = __float2bfloat16_rn(vd * v3 * silup_v1);
        dh13[row * row_stride + H + col] = __float2bfloat16_rn(vd * silu_v1);
    }
}

inline void launch(
    const __nv_bfloat16* dh,
    const __nv_bfloat16* h13,
    __nv_bfloat16* dh13,
    int64_t M,
    int64_t H,
    cudaStream_t stream
) {
    int64_t total_pairs = (M * H) / 2;
    int64_t grid = (total_pairs + THREADS - 1) / THREADS;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    silu_deriv_dual_interleaved_kernel<<<(int)grid, THREADS, 0, stream>>>(
        dh, h13, dh13, M, H);
}

}  // namespace tk_silu_deriv_interleaved
