#pragma once

#include "kittens.cuh"
#include "mxfp4_launch_config.cuh"

using namespace kittens;

namespace mxfp4_silu_dgrad_quant_gemm {

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _QUANT_MODE = 1, bool _USE_SAVED_SIGMOID = false>
struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = mxfp4_launch::default_use_pdl;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 256;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = 2;
    static constexpr int MMA_PER_TILE = 2;
    static constexpr int NUM_D_TILES = 2;
    static constexpr int QUANT_MODE = _QUANT_MODE;
    static constexpr bool USE_SAVED_SIGMOID = _USE_SAVED_SIGMOID;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;
    using row_fp4_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, B_sc_tile>;
    using row_fp4_gl = gl<fp4e2m1_2, 1,  1, -1, -1, row_fp4_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;

    const bf16* h3;
    const bf16* h1_raw;
    const bf16* sig_h1;

    row_fp4_gl row_fp4;
    uint8_t* row_sc;
    uint8_t* col0_fp4;
    uint8_t* col0_sc;
    uint8_t* col1_fp4;
    uint8_t* col1_sc;
    int M;
    int H;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A[C::MMA_PER_TILE];
        B_sc_tile B[C::B_SC_SIZE * C::MMA_PER_TILE];
    };

    __host__ inline dim3 grid() const {
        const int num_row_blocks = M / C::Mb;
        const int num_col_blocks = H / C::Nb;
        int grid_size = min(num_row_blocks * num_col_blocks * C::CLUSTER_SIZE, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(grid_size);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

__device__ __forceinline__ uint8_t float_to_fp4(float val) {
    float aval = fabsf(val);
    uint8_t sign = ((__float_as_uint(val) >> 31) << 3);
    uint8_t enc =
        static_cast<uint8_t>(aval >= 0.25f) +
        static_cast<uint8_t>(aval >= 0.75f) +
        static_cast<uint8_t>(aval >= 1.25f) +
        static_cast<uint8_t>(aval >= 1.75f) +
        static_cast<uint8_t>(aval >= 2.5f) +
        static_cast<uint8_t>(aval >= 3.5f) +
        static_cast<uint8_t>(aval >= 5.0f);
    return sign | enc;
}

__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float rcp_scale) {
    uint8_t q0 = float_to_fp4(v0 * rcp_scale);
    uint8_t q1 = float_to_fp4(v1 * rcp_scale);
    return q0 | (q1 << 4);
}

__device__ __forceinline__ uint8_t float_to_e8m0_rn(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t val_u32 = __float_as_uint(val);
    uint8_t exponent = (val_u32 >> 23) & 0xFF;
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    constexpr uint32_t half = 1u << 22;
    bool round_up = (mantissa > half) || (mantissa == half && (exponent & 1));
    if (round_up && exponent < 0xFE) ++exponent;
    return exponent;
}

__device__ __forceinline__ uint8_t float_to_e8m0_ceil(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t u = __float_as_uint(val);
    uint8_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    if (mant > 0 && exp < 0xFE) ++exp;
    return exp;
}

__device__ __forceinline__ uint8_t float_to_e8m0_floor(float val) {
    if (val == 0.0f) return 0x00;
    uint32_t u = __float_as_uint(val);
    return (u >> 23) & 0xFF;
}

template <int MODE>
__device__ __forceinline__ uint8_t float_to_e8m0_dispatch(float val) {
    if constexpr (MODE == 1) {
        return float_to_e8m0_ceil(val);
    } else if constexpr (MODE == 2) {
        return float_to_e8m0_floor(val);
    } else {
        return float_to_e8m0_rn(val);
    }
}

__device__ __forceinline__ float exp2f_rcp_e8m0(uint8_t e8m0) {
    if (e8m0 == 0) return 0.0f;
    uint32_t bits = (uint32_t)(254 - e8m0) << 23;
    return __uint_as_float(bits);
}

template <typename C>
__device__ __noinline__ void quantize_rows_from_stage(
    const globals<C>& g,
    bf16_2 (*col_pairs)[33],
    int lane_id,
    int warp_row_base,
    int logical_col_start,
    int row_fp4_stride)
{
    const int local_row = lane_id;
    const int global_row = warp_row_base + local_row;
    bf16_2 cached_pairs[16];
    float row_amax = 0.0f;

    #pragma unroll
    for (int pair = 0; pair < 16; pair++) {
        const bf16_2 v = col_pairs[pair][local_row];
        cached_pairs[pair] = v;
        row_amax = fmaxf(row_amax, fabsf(__bfloat162float(v.x)));
        row_amax = fmaxf(row_amax, fabsf(__bfloat162float(v.y)));
    }

    const uint8_t row_e8m0 = (row_amax <= 1e-9f) ? 0 : float_to_e8m0_dispatch<C::QUANT_MODE>(row_amax);
    const float row_coeff = 6.0f * exp2f_rcp_e8m0(row_e8m0);
    uint64_t packed_lo = 0;
    uint64_t packed_hi = 0;
    #pragma unroll
    for (int pair = 0; pair < 8; pair++) {
        const bf16_2 v = cached_pairs[pair];
        packed_lo |= static_cast<uint64_t>(
            quantize_fp4_pair(__bfloat162float(v.x), __bfloat162float(v.y), row_coeff)) << (pair * 8);
    }
    #pragma unroll
    for (int pair = 0; pair < 8; pair++) {
        const bf16_2 v = cached_pairs[pair + 8];
        packed_hi |= static_cast<uint64_t>(
            quantize_fp4_pair(__bfloat162float(v.x), __bfloat162float(v.y), row_coeff)) << (pair * 8);
    }

    uint8_t* row_fp4_ptr = reinterpret_cast<uint8_t*>(g.row_fp4.raw_ptr);
    const int row_fp4_base = global_row * row_fp4_stride + logical_col_start / 2;
    *reinterpret_cast<uint64_t*>(&row_fp4_ptr[row_fp4_base + 0]) = packed_lo;
    *reinterpret_cast<uint64_t*>(&row_fp4_ptr[row_fp4_base + 8]) = packed_hi;

    const int row_ntk_total = (2 * g.H) / 128;
    const int sc_row_blk = global_row / 128;
    const int j_in_tile = global_row % 32;
    const int grp = (global_row % 128) / 32;
    const int sc_col_blk = logical_col_start / 128;
    const int base = (sc_row_blk * row_ntk_total + sc_col_blk) * 512 + j_in_tile * 16 + grp * 4;
    g.row_sc[base + (logical_col_start % 128) / 32] = row_e8m0;
}

template <typename C>
__device__ __noinline__ void quantize_cols_from_stage(
    const globals<C>& g,
    bf16_2 (*col_pairs)[33],
    uint8_t* col_fp4,
    uint8_t* col_sc,
    int lane_id,
    int warp_row_base,
    int col_start)
{
    const int local_col = lane_id;
    const int local_col_pair = local_col >> 1;
    const bool use_y = (local_col & 1) != 0;
    const int global_col = col_start + local_col;
    bf16_2 cached_pairs[16];
    float col_amax = 0.0f;

    #pragma unroll
    for (int pair = 0; pair < 16; pair++) {
        const int row0 = pair * 2;
        const bf16_2 v0 = col_pairs[local_col_pair][row0 + 0];
        const bf16_2 v1 = col_pairs[local_col_pair][row0 + 1];
        cached_pairs[pair] = use_y ? bf16_2{v0.y, v1.y} : bf16_2{v0.x, v1.x};
        col_amax = fmaxf(col_amax, fabsf(__bfloat162float(cached_pairs[pair].x)));
        col_amax = fmaxf(col_amax, fabsf(__bfloat162float(cached_pairs[pair].y)));
    }

    const uint8_t col_e8m0 = (col_amax <= 1e-9f) ? 0 : float_to_e8m0_dispatch<C::QUANT_MODE>(col_amax);
    const float col_coeff = 6.0f * exp2f_rcp_e8m0(col_e8m0);
    const int global_row_pair_base = warp_row_base / 2;
    const int col_fp4_stride = g.M / 2;
    uint64_t packed_col_lo = 0;
    uint64_t packed_col_hi = 0;
    #pragma unroll
    for (int pair = 0; pair < 16; pair++) {
        const uint64_t packed_pair = static_cast<uint64_t>(quantize_fp4_pair(
            __bfloat162float(cached_pairs[pair].x),
            __bfloat162float(cached_pairs[pair].y),
            col_coeff));
        if (pair < 8) packed_col_lo |= packed_pair << (pair * 8);
        else          packed_col_hi |= packed_pair << ((pair - 8) * 8);
    }
    *reinterpret_cast<uint64_t*>(&col_fp4[global_col * col_fp4_stride + global_row_pair_base + 0]) = packed_col_lo;
    *reinterpret_cast<uint64_t*>(&col_fp4[global_col * col_fp4_stride + global_row_pair_base + 8]) = packed_col_hi;

    const int m_kgroup = warp_row_base / 128;
    const int m_32_in_128 = (warp_row_base / 32) % 4;
    const int depth = global_col / 128;
    const int sr = global_col % 32;
    const int rr = (global_col / 32) % 4;
    const int chunk = depth * (g.M / 128) + m_kgroup;
    const int byte_idx = sr * 16 + rr * 4 + m_32_in_128;
    col_sc[chunk * 512 + byte_idx] = col_e8m0;
}

__device__ __forceinline__ bf16_2 load_bf16_pair(const bf16* ptr, int row, int cols, int col) {
    return *reinterpret_cast<const bf16_2*>(&ptr[static_cast<int64_t>(row) * cols + col]);
}

__device__ __forceinline__ void silu_deriv_pair(
    float dh_x,
    float dh_y,
    const bf16_2 h3_pair,
    const bf16_2 h1_pair,
    bf16_2& out0,
    bf16_2& out1)
{
    const float h3_x = __bfloat162float(h3_pair.x);
    const float h3_y = __bfloat162float(h3_pair.y);
    const float h1_x = __bfloat162float(h1_pair.x);
    const float h1_y = __bfloat162float(h1_pair.y);
    const float sig_x = 1.0f / (1.0f + __expf(-h1_x));
    const float sig_y = 1.0f / (1.0f + __expf(-h1_y));
    const float silu_x = h1_x * sig_x;
    const float silu_y = h1_y * sig_y;
    const float silup_x = sig_x * (1.0f + h1_x - silu_x);
    const float silup_y = sig_y * (1.0f + h1_y - silu_y);
    out0 = bf16_2{
        __float2bfloat16_rn(dh_x * h3_x * silup_x),
        __float2bfloat16_rn(dh_y * h3_y * silup_y)};
    out1 = bf16_2{
        __float2bfloat16_rn(dh_x * silu_x),
        __float2bfloat16_rn(dh_y * silu_y)};
}

__device__ __forceinline__ void silu_deriv_pair_from_sigmoid(
    float dh_x,
    float dh_y,
    const bf16_2 h3_pair,
    const bf16_2 h1_pair,
    const bf16_2 sig_pair,
    bf16_2& out0,
    bf16_2& out1)
{
    const float h3_x = __bfloat162float(h3_pair.x);
    const float h3_y = __bfloat162float(h3_pair.y);
    const float h1_x = __bfloat162float(h1_pair.x);
    const float h1_y = __bfloat162float(h1_pair.y);
    const float sig_x = __bfloat162float(sig_pair.x);
    const float sig_y = __bfloat162float(sig_pair.y);
    const float silu_x = h1_x * sig_x;
    const float silu_y = h1_y * sig_y;
    const float silup_x = sig_x * (1.0f + h1_x * (1.0f - sig_x));
    const float silup_y = sig_y * (1.0f + h1_y * (1.0f - sig_y));
    out0 = bf16_2{
        __float2bfloat16_rn(dh_x * h3_x * silup_x),
        __float2bfloat16_rn(dh_y * h3_y * silup_y)};
    out1 = bf16_2{
        __float2bfloat16_rn(dh_x * silu_x),
        __float2bfloat16_rn(dh_y * silu_y)};
}

template <typename C>
__device__ __forceinline__ void silu_deriv_pair_dispatch(
    const globals<C>& g,
    float dh_x,
    float dh_y,
    const bf16_2 h3_pair,
    const bf16_2 h1_pair,
    int row,
    int col,
    bf16_2& out0,
    bf16_2& out1)
{
    if constexpr (C::USE_SAVED_SIGMOID) {
        const bf16_2 sig_pair = load_bf16_pair(g.sig_h1, row, g.H, col);
        silu_deriv_pair_from_sigmoid(dh_x, dh_y, h3_pair, h1_pair, sig_pair, out0, out1);
    } else {
        silu_deriv_pair(dh_x, dh_y, h3_pair, h1_pair, out0, out1);
    }
}

template <typename C, typename subtile_rt>
__device__ __noinline__ void stage_silu_deriv_pairs(
    const globals<C>& g,
    subtile_rt& D_fl,
    bf16_2 (*pairs0)[33],
    bf16_2 (*pairs1)[33],
    int lane_id,
    int warp_row_base,
    int col_start)
{
    constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
    const int lane_byte = lane_id % 4;
    const int row_pair_idx = lane_id / 4;

    #pragma unroll
    for (int i = 0; i < subtile_rt::height; i++) {
        const int row_lo = warp_row_base + i * 16 + row_pair_idx;
        const int row_hi = row_lo + 8;
        #pragma unroll
        for (int j = 0; j < subtile_rt::width; j++) {
            const int pair_base = j * 8 + lane_byte;
            const int pair_col0 = col_start + 2 * pair_base;
            const int pair_col1 = col_start + 2 * (pair_base + 4);

            bf16_2 h3_0 = load_bf16_pair(g.h3, row_lo, g.H, pair_col0);
            bf16_2 h1_0 = load_bf16_pair(g.h1_raw, row_lo, g.H, pair_col0);
            bf16_2 h3_1 = load_bf16_pair(g.h3, row_hi, g.H, pair_col0);
            bf16_2 h1_1 = load_bf16_pair(g.h1_raw, row_hi, g.H, pair_col0);
            bf16_2 h3_2 = load_bf16_pair(g.h3, row_lo, g.H, pair_col1);
            bf16_2 h1_2 = load_bf16_pair(g.h1_raw, row_lo, g.H, pair_col1);
            bf16_2 h3_3 = load_bf16_pair(g.h3, row_hi, g.H, pair_col1);
            bf16_2 h1_3 = load_bf16_pair(g.h1_raw, row_hi, g.H, pair_col1);

            bf16_2 out0, out1;
            silu_deriv_pair_dispatch<C>(
                g,
                D_fl.tiles[i][j].data[0].x * MXFP4_ALPHA,
                D_fl.tiles[i][j].data[0].y * MXFP4_ALPHA,
                h3_0, h1_0, row_lo, pair_col0, out0, out1);
            pairs0[pair_base][i * 16 + row_pair_idx] = out0;
            pairs1[pair_base][i * 16 + row_pair_idx] = out1;

            silu_deriv_pair_dispatch<C>(
                g,
                D_fl.tiles[i][j].data[1].x * MXFP4_ALPHA,
                D_fl.tiles[i][j].data[1].y * MXFP4_ALPHA,
                h3_1, h1_1, row_hi, pair_col0, out0, out1);
            pairs0[pair_base][i * 16 + row_pair_idx + 8] = out0;
            pairs1[pair_base][i * 16 + row_pair_idx + 8] = out1;

            silu_deriv_pair_dispatch<C>(
                g,
                D_fl.tiles[i][j].data[2].x * MXFP4_ALPHA,
                D_fl.tiles[i][j].data[2].y * MXFP4_ALPHA,
                h3_2, h1_2, row_lo, pair_col1, out0, out1);
            pairs0[pair_base + 4][i * 16 + row_pair_idx] = out0;
            pairs1[pair_base + 4][i * 16 + row_pair_idx] = out1;

            silu_deriv_pair_dispatch<C>(
                g,
                D_fl.tiles[i][j].data[3].x * MXFP4_ALPHA,
                D_fl.tiles[i][j].data[3].y * MXFP4_ALPHA,
                h3_3, h1_3, row_hi, pair_col1, out0, out1);
            pairs0[pair_base + 4][i * 16 + row_pair_idx + 8] = out0;
            pairs1[pair_base + 4][i * 16 + row_pair_idx + 8] = out1;
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C>& g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.M / C::Mb;
    const int num_col_blocks = g.H / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_iters_per_block = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        tma::cluster::load_async(input_scales[stage].A[k], g.A_sc,
                            {row_block_idx*2 + cta_id, i*C::MMA_PER_TILE + k, 0, 0},
                            scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    }
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        tma::cluster::load_async(
                            input_scales[stage].B[cta_id * C::MMA_PER_TILE + k], g.B_sc,
                            {col_block_idx*2 + cta_id, i*C::MMA_PER_TILE + k, 0, 0},
                            scales_arrived[stage], (uint16_t)(0b11), 0);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            uint32_t output_phasebits = 0xFFFF0000;
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                wait(outputs_finished, get_phasebit<1>(output_phasebits, 0));
                tensor_after_thread_sync();
                bool issued_mma = false;
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*16 + k*16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*C::B_SC_SIZE*16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                        auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*C::B_SC_SIZE*16 + 16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[C::MMA_PER_TILE + k]);
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (!issued_mma) {
                        mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                            A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]);
                        issued_mma = true;
                    } else {
                        mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                            A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]);
                    }
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(output_phasebits, 0);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        if constexpr (C::USE_PDL) warpgroup::pdl::wait();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        static_assert(SUBTILE_COLS == 32, "SiLU dgrad producer assumes 32-column epilogue slices");
        __shared__ bf16_2 col_stage_smem0[WARPGROUP_WARPS][16][33];
        __shared__ bf16_2 col_stage_smem1[WARPGROUP_WARPS][16][33];
        uint32_t output_phasebits = 0xFFFF0000;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            const int warp_row_base = (row_block_idx * 2 + cta_id) * (C::Mb / 2) + warpgroup::warpid() * 32;

            wait(outputs_arrived, get_phasebit<0>(output_phasebits, 0));

            subtile_rt D_regs_fl[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                warpgroup::load_async(
                    D_regs_fl[epi],
                    out_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            const int row_fp4_stride = g.row_fp4.cols();
            const int lane_id = warp::laneid();
            bf16_2 (*pairs0)[33] = col_stage_smem0[warpgroup::warpid()];
            bf16_2 (*pairs1)[33] = col_stage_smem1[warpgroup::warpid()];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                stage_silu_deriv_pairs<C>(
                    g, D_regs_fl[epi], pairs0, pairs1, lane_id, warp_row_base, col_start);
                __syncwarp();
                quantize_rows_from_stage<C>(
                    g, pairs0, lane_id, warp_row_base, col_start, row_fp4_stride);
                quantize_cols_from_stage<C>(
                    g, pairs0, g.col0_fp4, g.col0_sc, lane_id, warp_row_base, col_start);
                quantize_rows_from_stage<C>(
                    g, pairs1, lane_id, warp_row_base, g.H + col_start, row_fp4_stride);
                quantize_cols_from_stage<C>(
                    g, pairs1, g.col1_fp4, g.col1_sc, lane_id, warp_row_base, col_start);
            }
            update_phasebit<0>(output_phasebits, 0);
        }
        warpgroup::sync(1);
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
    }

    // Keep both clustered CTAs resident until peer DSM/TMEM tail traffic retires.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

} // namespace mxfp4_silu_dgrad_quant_gemm
