#pragma once

#include "kittens.cuh"

namespace h_mxfp4_tile_carrier {

constexpr int TILE = 128;
constexpr float TILE_INV_ELEMENTS = 0x1p-14f;

// MX mode=1: encode-centric E8M0. The scale always encloses amax.
__device__ __forceinline__ uint8_t e8m0_mode1(float amax) {
    if (!(amax > 0.0f)) return 0;
    uint32_t u = __float_as_uint(amax);
    uint8_t e = static_cast<uint8_t>((u >> 23) & 0xffu);
    if ((u & 0x7fffffu) != 0 && e < 0xfeu) ++e;
    return e;
}

__device__ __forceinline__ float fp4_rcp(uint8_t e) {
    if (e == 0) return 0.0f;
    return 6.0f * __uint_as_float(static_cast<uint32_t>(254 - e) << 23);
}

__device__ __forceinline__ uint32_t pack_fp4_8(
    uint64_t in03, uint64_t in47, float coeff) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg.b64 coeff2;\n\t"
        "mov.b64 coeff2, {%3, %3};\n\t"
        ".reg.b16 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
        "mov.b64 {b0, b1, b2, b3}, %1;\n\t"
        "mov.b64 {b4, b5, b6, b7}, %2;\n\t"
        ".reg.b32 f0, f1, f2, f3, f4, f5, f6, f7;\n\t"
        "cvt.f32.bf16 f0, b0;\n\t"
        "cvt.f32.bf16 f1, b1;\n\t"
        "cvt.f32.bf16 f2, b2;\n\t"
        "cvt.f32.bf16 f3, b3;\n\t"
        "cvt.f32.bf16 f4, b4;\n\t"
        "cvt.f32.bf16 f5, b5;\n\t"
        "cvt.f32.bf16 f6, b6;\n\t"
        "cvt.f32.bf16 f7, b7;\n\t"
        ".reg.b64 f01, f23, f45, f67;\n\t"
        "mov.b64 f01, {f0, f1};\n\t"
        "mov.b64 f23, {f2, f3};\n\t"
        "mov.b64 f45, {f4, f5};\n\t"
        "mov.b64 f67, {f6, f7};\n\t"
        "mul.f32x2 f01, f01, coeff2;\n\t"
        "mul.f32x2 f23, f23, coeff2;\n\t"
        "mul.f32x2 f45, f45, coeff2;\n\t"
        "mul.f32x2 f67, f67, coeff2;\n\t"
        "mov.b64 {f1, f0}, f01;\n\t"
        "mov.b64 {f3, f2}, f23;\n\t"
        "mov.b64 {f5, f4}, f45;\n\t"
        "mov.b64 {f7, f6}, f67;\n\t"
        ".reg.b8 q0, q1, q2, q3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q0, f0, f1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q1, f2, f3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q2, f4, f5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q3, f6, f7;\n\t"
        "mov.b32 %0, {q0, q1, q2, q3};\n\t"
        "}"
        : "=r"(out)
        : "l"(in03), "l"(in47), "f"(coeff));
    return out;
}

__device__ __forceinline__ uint4 pack_fp4_32(
    const kittens::bf16_2 (&cached)[16], float coeff) {
    uint4 out;
    out.x = pack_fp4_8(
        *reinterpret_cast<const uint64_t*>(&cached[0]),
        *reinterpret_cast<const uint64_t*>(&cached[2]), coeff);
    out.y = pack_fp4_8(
        *reinterpret_cast<const uint64_t*>(&cached[4]),
        *reinterpret_cast<const uint64_t*>(&cached[6]), coeff);
    out.z = pack_fp4_8(
        *reinterpret_cast<const uint64_t*>(&cached[8]),
        *reinterpret_cast<const uint64_t*>(&cached[10]), coeff);
    out.w = pack_fp4_8(
        *reinterpret_cast<const uint64_t*>(&cached[12]),
        *reinterpret_cast<const uint64_t*>(&cached[14]), coeff);
    return out;
}

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

__device__ __forceinline__ float tile_rsqrt(float s, float* scratch, float eps) {
    const int lane = kittens::warp::laneid();
    const int warp = kittens::warpgroup::warpid();
    #pragma unroll
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffffu, s, d);
    if (lane == 0) scratch[warp] = s;
    kittens::warpgroup::sync(1);
    if (warp == 0) {
        float t = lane < kittens::WARPGROUP_WARPS ? scratch[lane] : 0.0f;
        #pragma unroll
        for (int d = 16; d; d >>= 1) t += __shfl_down_sync(0xffffffffu, t, d);
        if (lane == 0) {
            scratch[kittens::WARPGROUP_WARPS] =
                rsqrtf(fmaf(t, TILE_INV_ELEMENTS, eps));
        }
    }
    kittens::warpgroup::sync(1);
    return scratch[kittens::WARPGROUP_WARPS];
}

template <typename RT>
__device__ __forceinline__ void normalize_gamma_to_stage(
    const RT& z,
    kittens::bf16_2 (*pairs)[33],
    float r,
    const kittens::bf16* gamma,
    int col_base) {
    const int lane = kittens::warp::laneid();
    const int lane_byte = lane & 3;
    const int row_pair = lane >> 2;
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const int p = j * 8 + lane_byte;
            const int c0 = col_base + p * 2;
            const int c1 = c0 + 8;
            const auto& q = z.tiles[i][j].data;
            pairs[p][i * 16 + row_pair] = kittens::bf16_2{
                __float2bfloat16_rn(
                    (__bfloat162float(q[0].x) * r) *
                    __bfloat162float(gamma[c0])),
                __float2bfloat16_rn(
                    (__bfloat162float(q[0].y) * r) *
                    __bfloat162float(gamma[c0 + 1]))};
            pairs[p][i * 16 + row_pair + 8] = kittens::bf16_2{
                __float2bfloat16_rn(
                    (__bfloat162float(q[1].x) * r) *
                    __bfloat162float(gamma[c0])),
                __float2bfloat16_rn(
                    (__bfloat162float(q[1].y) * r) *
                    __bfloat162float(gamma[c0 + 1]))};
            pairs[p + 4][i * 16 + row_pair] = kittens::bf16_2{
                __float2bfloat16_rn(
                    (__bfloat162float(q[2].x) * r) *
                    __bfloat162float(gamma[c1])),
                __float2bfloat16_rn(
                    (__bfloat162float(q[2].y) * r) *
                    __bfloat162float(gamma[c1 + 1]))};
            pairs[p + 4][i * 16 + row_pair + 8] = kittens::bf16_2{
                __float2bfloat16_rn(
                    (__bfloat162float(q[3].x) * r) *
                    __bfloat162float(gamma[c1])),
                __float2bfloat16_rn(
                    (__bfloat162float(q[3].y) * r) *
                    __bfloat162float(gamma[c1 + 1]))};
        }
    }
}

template <typename G>
__device__ __forceinline__ void emit_32x32(
    const G& g,
    kittens::bf16_2 (*pairs)[33],
    int warp_row,
    int col_start) {
    const int lane = kittens::warp::laneid();
    const int row = warp_row + lane;
    kittens::bf16_2 cached[16];
    float amax = 0.0f;
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        cached[p] = pairs[p][lane];
        amax = fmaxf(amax, fabsf(__bfloat162float(cached[p].x)));
        amax = fmaxf(amax, fabsf(__bfloat162float(cached[p].y)));
    }
    const uint8_t e = e8m0_mode1(amax);
    const float rcp = fp4_rcp(e);
    auto* row_ptr = reinterpret_cast<uint8_t*>(g.h_row_fp4);
    *reinterpret_cast<uint4*>(
        row_ptr + row * (g.h_cols / 2) + col_start / 2) =
        pack_fp4_32(cached, rcp);
    const int rsb = row / 128;
    const int jig = row % 32;
    const int grp = (row % 128) / 32;
    g.h_row_sc[
        ((rsb * (g.h_cols / 128) + col_start / 128) * 512) +
        jig * 16 + grp * 4 + (col_start % 128) / 32] = e;

    const int local_col = lane;
    const int pair_col = local_col >> 1;
    const bool y = local_col & 1;
    float camax = 0.0f;
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        const auto a = pairs[pair_col][p * 2];
        const auto b = pairs[pair_col][p * 2 + 1];
        cached[p] = y ? kittens::bf16_2{a.y, b.y}
                      : kittens::bf16_2{a.x, b.x};
        camax = fmaxf(camax, fabsf(__bfloat162float(cached[p].x)));
        camax = fmaxf(camax, fabsf(__bfloat162float(cached[p].y)));
    }
    const uint8_t ce = e8m0_mode1(camax);
    const float crcp = fp4_rcp(ce);
    const int gc = col_start + local_col;
    auto* col_ptr = reinterpret_cast<uint8_t*>(g.h_col_fp4);
    *reinterpret_cast<uint4*>(
        col_ptr + gc * (g.h_rows / 2) + warp_row / 2) =
        pack_fp4_32(cached, crcp);
    const int chunk = (gc / 128) * (g.h_rows / 128) + warp_row / 128;
    const int idx =
        (gc % 32) * 16 + ((gc / 32) % 4) * 4 + ((warp_row / 32) % 4);
    g.h_col_sc[chunk * 512 + idx] = ce;
}

}  // namespace h_mxfp4_tile_carrier
