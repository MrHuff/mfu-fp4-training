#pragma once

namespace mxfp4_rope_epilogue {

struct rope_desc {
    const float* cos = nullptr;
    const float* sin = nullptr;
    int seq_len = 0;
    int head_dim = 0;
    int rotary_dim = 0;

    __host__ __device__ inline bool enabled() const {
        return cos != nullptr && sin != nullptr && seq_len > 0 && head_dim > 0 && rotary_dim > 0;
    }

    __host__ __device__ inline int pair_dim() const {
        return rotary_dim / 2;
    }
};

struct rope_live64_desc {
    const float2* cs = nullptr;
    int seq_len = 0;
    int seq_mask = 0;
    int pair_stride = 32;
    int head_mask = 63;

    __host__ __device__ inline bool enabled() const {
        return cs != nullptr && seq_len > 0;
    }
};

template <typename Packed>
__device__ inline Packed rotate_pair(const Packed& packed, float cos_v, float sin_v);

template <>
__device__ inline float2 rotate_pair<float2>(const float2& packed, float cos_v, float sin_v) {
    return float2{
        packed.x * cos_v - packed.y * sin_v,
        packed.y * cos_v + packed.x * sin_v,
    };
}

template <>
__device__ inline kittens::bf16_2 rotate_pair<kittens::bf16_2>(const kittens::bf16_2& packed, float cos_v, float sin_v) {
    const float even = __bfloat162float(packed.x);
    const float odd = __bfloat162float(packed.y);
    return kittens::bf16_2{
        __float2bfloat16_rn(even * cos_v - odd * sin_v),
        __float2bfloat16_rn(odd * cos_v + even * sin_v),
    };
}

template <typename Packed>
__device__ inline Packed inverse_rotate_pair(const Packed& packed, float cos_v, float sin_v);

template <>
__device__ inline float2 inverse_rotate_pair<float2>(const float2& packed, float cos_v, float sin_v) {
    return float2{
        packed.x * cos_v + packed.y * sin_v,
        packed.y * cos_v - packed.x * sin_v,
    };
}

template <>
__device__ inline kittens::bf16_2 inverse_rotate_pair<kittens::bf16_2>(const kittens::bf16_2& packed, float cos_v, float sin_v) {
    const float even = __bfloat162float(packed.x);
    const float odd = __bfloat162float(packed.y);
    return kittens::bf16_2{
        __float2bfloat16_rn(even * cos_v + odd * sin_v),
        __float2bfloat16_rn(odd * cos_v - even * sin_v),
    };
}

template <typename Packed>
__device__ inline float2 unpack_pair(const Packed& packed);

template <>
__device__ inline float2 unpack_pair<float2>(const float2& packed) {
    return packed;
}

template <>
__device__ inline float2 unpack_pair<kittens::bf16_2>(const kittens::bf16_2& packed) {
    return float2{__bfloat162float(packed.x), __bfloat162float(packed.y)};
}

template <typename Packed>
__device__ inline Packed pack_pair(float x, float y);

template <>
__device__ inline float2 pack_pair<float2>(float x, float y) {
    return float2{x, y};
}

template <>
__device__ inline kittens::bf16_2 pack_pair<kittens::bf16_2>(float x, float y) {
    return kittens::bf16_2{__float2bfloat16_rn(x), __float2bfloat16_rn(y)};
}

template <typename Packed>
__device__ inline void load_rope_stride1_live64(
    float (&vals)[8],
    int dst,
    const Packed& packed,
    const rope_live64_desc& rope,
    int row,
    int col_even
) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    const float2 cs = rope.cs[
        (row & rope.seq_mask) * rope.pair_stride +
        ((col_even & rope.head_mask) >> 1)
    ];
    const float2 v = unpack_pair(packed);
    const float r0 = v.x * cs.x - v.y * cs.y;
    const float r1 = v.y * cs.x + v.x * cs.y;
    vals[dst + 0] = (r0 + r1) * inv_sqrt2;
    vals[dst + 1] = (r0 - r1) * inv_sqrt2;
}

__device__ inline void scalar_butterfly(float& lo, float& hi) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    const float a = lo;
    const float b = hi;
    lo = (a + b) * inv_sqrt2;
    hi = (a - b) * inv_sqrt2;
}

__device__ inline void hadamard32_distributed_4lane(float (&vals)[8], int warp_lane) {
    constexpr float inv_sqrt2 = 0.70710678118654752440f;

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float peer = __shfl_xor_sync(0xffffffff, vals[e], 1);
        vals[e] = ((warp_lane & 1) == 0)
            ? (vals[e] + peer) * inv_sqrt2
            : (peer - vals[e]) * inv_sqrt2;
    }

    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float peer = __shfl_xor_sync(0xffffffff, vals[e], 2);
        vals[e] = ((warp_lane & 2) == 0)
            ? (vals[e] + peer) * inv_sqrt2
            : (peer - vals[e]) * inv_sqrt2;
    }

    scalar_butterfly(vals[0], vals[2]);
    scalar_butterfly(vals[1], vals[3]);
    scalar_butterfly(vals[4], vals[6]);
    scalar_butterfly(vals[5], vals[7]);
    scalar_butterfly(vals[0], vals[4]);
    scalar_butterfly(vals[1], vals[5]);
    scalar_butterfly(vals[2], vals[6]);
    scalar_butterfly(vals[3], vals[7]);
}

template <kittens::ducks::rt::row_layout RT>
__device__ inline void apply_inplace_live64_rht32(
    RT& tile,
    const rope_live64_desc& rope,
    int global_row_base,
    int global_col_base
) {
    if (!rope.enabled()) {
        return;
    }

    static_assert(
        std::is_same_v<typename RT::dtype, float2> || std::is_same_v<typename RT::dtype, kittens::bf16_2>,
        "RoPE+RHT epilogue currently supports float2 and bf16_2 register tiles only"
    );

    constexpr int tile_row_dim = RT::tile_size_row;
    constexpr int tile_col_dim = RT::tile_size_col;
    const int warp_row_base = kittens::warpgroup::warpid() * RT::rows;
    const int warp_lane = kittens::warp::laneid();
    const int lane_pair_col = (warp_lane % 4) * 2;

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j + 1 < RT::width; j += 2) {
            #pragma unroll
            for (int row_half = 0; row_half < 2; ++row_half) {
                const int k0 = row_half;
                const int k8 = row_half + 2;
                const int row =
                    global_row_base +
                    warp_row_base +
                    i * tile_row_dim +
                    row_half * (tile_row_dim / 2) +
                    warp_lane / 4;
                const int col_base = global_col_base + j * tile_col_dim + lane_pair_col;

                float vals[8];
                load_rope_stride1_live64(vals, 0, tile.tiles[i][j + 0].data[k0], rope, row, col_base + 0);
                load_rope_stride1_live64(vals, 2, tile.tiles[i][j + 0].data[k8], rope, row, col_base + 8);
                load_rope_stride1_live64(vals, 4, tile.tiles[i][j + 1].data[k0], rope, row, col_base + 16);
                load_rope_stride1_live64(vals, 6, tile.tiles[i][j + 1].data[k8], rope, row, col_base + 24);

                hadamard32_distributed_4lane(vals, warp_lane);

                tile.tiles[i][j + 0].data[k0] = pack_pair<typename RT::dtype>(vals[0], vals[1]);
                tile.tiles[i][j + 0].data[k8] = pack_pair<typename RT::dtype>(vals[2], vals[3]);
                tile.tiles[i][j + 1].data[k0] = pack_pair<typename RT::dtype>(vals[4], vals[5]);
                tile.tiles[i][j + 1].data[k8] = pack_pair<typename RT::dtype>(vals[6], vals[7]);
            }
        }
    }
}

template <kittens::ducks::rt::row_layout RT>
__device__ inline void apply_inplace(
    RT& tile,
    const rope_desc& rope,
    int global_row_base,
    int global_col_base
) {
    if (!rope.enabled()) {
        return;
    }

    static_assert(
        std::is_same_v<typename RT::dtype, float2> || std::is_same_v<typename RT::dtype, kittens::bf16_2>,
        "RoPE epilogue currently supports float2 and bf16_2 register tiles only"
    );

    constexpr int tile_row_dim = RT::tile_size_row;
    constexpr int tile_col_dim = RT::tile_size_col;
    const int warp_row_base = kittens::warpgroup::warpid() * RT::rows;
    const int warp_lane = kittens::warp::laneid();

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; ++k) {
                const int row =
                    global_row_base +
                    warp_row_base +
                    i * tile_row_dim +
                    (k % 2) * (tile_row_dim / 2) +
                    warp_lane / 4;
                const int col_even =
                    global_col_base +
                    j * tile_col_dim +
                    (k / 2) * (tile_col_dim / 2) +
                    (warp_lane % 4) * 2;
                const int head_col = col_even % rope.head_dim;
                if (head_col < rope.rotary_dim) {
                    const int rope_offset = (row % rope.seq_len) * rope.pair_dim() + (head_col / 2);
                    tile.tiles[i][j].data[k] = rotate_pair(tile.tiles[i][j].data[k], rope.cos[rope_offset], rope.sin[rope_offset]);
                }
            }
        }
    }
}

template <kittens::ducks::rt::row_layout RT>
__device__ inline void apply_inplace_live64(
    RT& tile,
    const rope_live64_desc& rope,
    int global_row_base,
    int global_col_base
) {
    if (!rope.enabled()) {
        return;
    }

    static_assert(
        std::is_same_v<typename RT::dtype, float2> || std::is_same_v<typename RT::dtype, kittens::bf16_2>,
        "RoPE epilogue currently supports float2 and bf16_2 register tiles only"
    );

    constexpr int tile_row_dim = RT::tile_size_row;
    constexpr int tile_col_dim = RT::tile_size_col;
    const int warp_row_base = kittens::warpgroup::warpid() * RT::rows;
    const int warp_lane = kittens::warp::laneid();

    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; ++k) {
                const int row =
                    global_row_base +
                    warp_row_base +
                    i * tile_row_dim +
                    (k % 2) * (tile_row_dim / 2) +
                    warp_lane / 4;
                const int col_even =
                    global_col_base +
                    j * tile_col_dim +
                    (k / 2) * (tile_col_dim / 2) +
                    (warp_lane % 4) * 2;
                const float2 cs = rope.cs[
                    (row & rope.seq_mask) * rope.pair_stride +
                    ((col_even & rope.head_mask) >> 1)
                ];
                tile.tiles[i][j].data[k] = rotate_pair(tile.tiles[i][j].data[k], cs.x, cs.y);
            }
        }
    }
}

} // namespace mxfp4_rope_epilogue
