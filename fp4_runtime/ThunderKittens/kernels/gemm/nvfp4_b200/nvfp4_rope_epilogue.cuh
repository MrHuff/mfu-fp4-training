#pragma once

#include <type_traits>

namespace nvfp4_rope_epilogue {

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
    int pair_dim = 32;
    int head_mask = 63;
    bool round_input_bf16 = false;

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
__device__ inline kittens::bf16_2 rotate_pair<kittens::bf16_2>(
    const kittens::bf16_2& packed,
    float cos_v,
    float sin_v
) {
    const float even = __bfloat162float(packed.x);
    const float odd = __bfloat162float(packed.y);
    return kittens::bf16_2{
        __float2bfloat16_rn(even * cos_v - odd * sin_v),
        __float2bfloat16_rn(odd * cos_v + even * sin_v),
    };
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
        "RoPE epilogue supports float2 and bf16_2 register tiles"
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
                    tile.tiles[i][j].data[k] = rotate_pair(
                        tile.tiles[i][j].data[k],
                        rope.cos[rope_offset],
                        rope.sin[rope_offset]
                    );
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
        "RoPE epilogue supports float2 and bf16_2 register tiles"
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
                    (row & rope.seq_mask) * rope.pair_dim +
                    ((col_even & rope.head_mask) >> 1)
                ];
                if constexpr (std::is_same_v<typename RT::dtype, float2>) {
                    if (rope.round_input_bf16) {
                        auto &value = tile.tiles[i][j].data[k];
                        value.x = __bfloat162float(__float2bfloat16_rn(value.x));
                        value.y = __bfloat162float(__float2bfloat16_rn(value.y));
                    }
                }
                tile.tiles[i][j].data[k] = rotate_pair(tile.tiles[i][j].data[k], cs.x, cs.y);
            }
        }
    }
}

} // namespace nvfp4_rope_epilogue
