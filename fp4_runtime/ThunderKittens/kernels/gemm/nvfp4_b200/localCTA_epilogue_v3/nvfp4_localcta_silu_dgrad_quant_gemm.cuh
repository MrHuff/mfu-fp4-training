#pragma once

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_localcta_silu_dgrad_quant_gemm {

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _USE_PDL = true>
struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = _USE_PDL;

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
    static constexpr int MMA_PER_TILE = 4;
    static constexpr int NUM_D_TILES = 2;
};

template <typename _GL>
struct tma_dev_proxy {
    using identifier = ducks::gl::identifier;
    using T = typename _GL::T;
    using T2 = typename _GL::T2;
    using dtype = typename _GL::dtype;
    static constexpr int __b__ = _GL::__b__;
    static constexpr int __d__ = _GL::__d__;
    static constexpr int __r__ = _GL::__r__;
    static constexpr int __c__ = _GL::__c__;

    const CUtensorMap* dev_tma;

    __device__ explicit tma_dev_proxy(const CUtensorMap* _dev_tma) : dev_tma(_dev_tma) {}

    template<int axis> __device__ inline size_t shape() const { return 0; }
    template<int axis> __device__ inline size_t stride() const { return 0; }

    template<typename U, int axis> __device__ inline const CUtensorMap* get_tma() const {
        return dev_tma;
    }
    __device__ inline void prefetch() const {
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(dev_tma)) : "memory");
    }
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<half,       1, -1, -1, 256, B_sc_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;
    const float* A_sg;
    int A_sg_stride;
    const float* B_sg;
    int B_sg_stride;

    const bf16* h3;
    const bf16* h1_raw;

    uint8_t* row_fp4;
    uint8_t* row_sc;
    float* row_sg;
    uint8_t* col_fp4;
    uint8_t* col_sc;
    float* col_sg;
    int M;
    int H;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };

    __host__ inline dim3 grid() const {
        const int num_row_blocks = M / C::Mb;
        const int num_col_blocks = H / C::Nb;
        int grid_size = min(num_row_blocks * num_col_blocks * C::CLUSTER_SIZE, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

static constexpr float LOCALCTA_GLOBAL_SCALE_NUM = 1493.0f;
static constexpr float LOCALCTA_MIN_NONZERO_SCALE = 0.001953125f;
static constexpr float FP8_E4M3_MAX = 448.0f;

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

__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float coeff) {
    uint8_t q0 = float_to_fp4(v0 * coeff);
    uint8_t q1 = float_to_fp4(v1 * coeff);
    return q0 | (q1 << 4);
}

__device__ __forceinline__ uint8_t fp8e4m3_byte(float value) {
    fp8e4m3 out = static_cast<fp8e4m3>(value);
    return reinterpret_cast<const uint8_t&>(out);
}

__device__ __forceinline__ float fp8e4m3_round_to_float(float value) {
    fp8e4m3 out = static_cast<fp8e4m3>(value);
    return static_cast<float>(out);
}

__device__ __forceinline__ float localcta_encode_scale(float amax) {
    if (amax <= 1.0e-9f) {
        return 1.0f;
    }
    return fminf(LOCALCTA_GLOBAL_SCALE_NUM / amax, 3.4028235e+38f);
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ void localcta_block_quant_params(
    float block_amax,
    float chunk_s_enc,
    float chunk_sg,
    float& coeff,
    uint8_t& stored_scale_byte)
{
    float mult = FP8_E4M3_MAX;
    if (block_amax > 1.0e-9f && chunk_s_enc > 0.0f) {
        mult = fminf(6.0f / (block_amax * chunk_s_enc), 3.4028235e+38f);
    }
    const float mult_fp8 = fmaxf(fp8e4m3_round_to_float(mult), 1.0e-12f);
    coeff = mult_fp8 * chunk_s_enc;
    float stored = fp8e4m3_round_to_float(1.0f / mult_fp8) * chunk_sg;
    uint8_t byte = fp8e4m3_byte(stored);
    if (stored > 0.0f && byte == 0) {
        byte = fp8e4m3_byte(LOCALCTA_MIN_NONZERO_SCALE);
    }
    stored_scale_byte = byte;
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

template <typename C, typename subtile_rt>
__device__ __noinline__ void stage_silu_deriv_pairs(
    const globals<C>& g,
    subtile_rt& D_fl,
    bf16_2 (*pairs0)[16][33],
    bf16_2 (*pairs1)[16][33],
    int epi_slot,
    int lane_id,
    int warp_row_base,
    int col_start,
    float& local_amax0,
    float& local_amax1)
{
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
            silu_deriv_pair(
                D_fl.tiles[i][j].data[0].x,
                D_fl.tiles[i][j].data[0].y,
                h3_0, h1_0, out0, out1);
            pairs0[epi_slot][pair_base][i * 16 + row_pair_idx] = out0;
            pairs1[epi_slot][pair_base][i * 16 + row_pair_idx] = out1;
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.x)));
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.y)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.x)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.y)));

            silu_deriv_pair(
                D_fl.tiles[i][j].data[1].x,
                D_fl.tiles[i][j].data[1].y,
                h3_1, h1_1, out0, out1);
            pairs0[epi_slot][pair_base][i * 16 + row_pair_idx + 8] = out0;
            pairs1[epi_slot][pair_base][i * 16 + row_pair_idx + 8] = out1;
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.x)));
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.y)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.x)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.y)));

            silu_deriv_pair(
                D_fl.tiles[i][j].data[2].x,
                D_fl.tiles[i][j].data[2].y,
                h3_2, h1_2, out0, out1);
            pairs0[epi_slot][pair_base + 4][i * 16 + row_pair_idx] = out0;
            pairs1[epi_slot][pair_base + 4][i * 16 + row_pair_idx] = out1;
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.x)));
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.y)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.x)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.y)));

            silu_deriv_pair(
                D_fl.tiles[i][j].data[3].x,
                D_fl.tiles[i][j].data[3].y,
                h3_3, h1_3, out0, out1);
            pairs0[epi_slot][pair_base + 4][i * 16 + row_pair_idx + 8] = out0;
            pairs1[epi_slot][pair_base + 4][i * 16 + row_pair_idx + 8] = out1;
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.x)));
            local_amax0 = fmaxf(local_amax0, fabsf(__bfloat162float(out0.y)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.x)));
            local_amax1 = fmaxf(local_amax1, fabsf(__bfloat162float(out1.y)));
        }
    }
}

template <typename C>
__device__ __noinline__ void quantize_rows_from_stage(
    const globals<C>& g,
    bf16_2 (*pairs)[16][33],
    int epi_slot,
    int lane_id,
    int warp_row_base,
    int logical_col_start,
    float chunk_s_enc,
    float chunk_sg)
{
    const int local_row = lane_id;
    const int global_row = warp_row_base + local_row;
    const int row_fp4_stride = g.H;
    const int row_ntk_total = (2 * g.H) / 64;
    const int sc_row_blk = global_row / 128;
    const int j_in_tile = global_row % 32;
    const int grp = (global_row % 128) / 32;

    #pragma unroll
    for (int group = 0; group < 2; ++group) {
        float block_amax = 0.0f;
        bf16_2 cached[8];
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const bf16_2 v = pairs[epi_slot][group * 8 + pair][local_row];
            cached[pair] = v;
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(v.x)));
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(v.y)));
        }

        float coeff;
        uint8_t stored_scale;
        localcta_block_quant_params(block_amax, chunk_s_enc, chunk_sg, coeff, stored_scale);

        uint64_t packed = 0;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const bf16_2 v = cached[pair];
            packed |= static_cast<uint64_t>(quantize_fp4_pair(
                __bfloat162float(v.x), __bfloat162float(v.y), coeff)) << (pair * 8);
        }

        const int logical_col = logical_col_start + group * 16;
        *reinterpret_cast<uint64_t*>(
            &g.row_fp4[global_row * row_fp4_stride + logical_col / 2]) = packed;

        const int col64 = logical_col / 64;
        const int scale_in64 = (logical_col % 64) / 16;
        const int scale_base = (sc_row_blk * row_ntk_total + col64) * 512 +
                               j_in_tile * 16 + grp * 4 + scale_in64;
        g.row_sc[scale_base] = stored_scale;
    }
}

template <typename C>
__device__ __noinline__ void quantize_cols_from_stage(
    const globals<C>& g,
    bf16_2 (*pairs)[16][33],
    int epi_slot,
    int lane_id,
    int warp_row_base,
    int logical_col_start,
    float chunk_s_enc,
    float chunk_sg)
{
    const int local_col = lane_id;
    const int local_col_pair = local_col >> 1;
    const bool use_y = (local_col & 1) != 0;
    const int logical_col = logical_col_start + local_col;
    const int col_fp4_stride = g.M / 2;
    const int col_ntk_total = g.M / 64;
    const int col_chunk = logical_col / 128;
    const int j_in_tile = logical_col % 32;
    const int grp = (logical_col % 128) / 32;

    #pragma unroll
    for (int row_group = 0; row_group < 2; ++row_group) {
        float block_amax = 0.0f;
        bf16_2 cached[8];
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const int row0 = row_group * 16 + pair * 2;
            const bf16_2 v0 = pairs[epi_slot][local_col_pair][row0 + 0];
            const bf16_2 v1 = pairs[epi_slot][local_col_pair][row0 + 1];
            cached[pair] = use_y ? bf16_2{v0.y, v1.y} : bf16_2{v0.x, v1.x};
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(cached[pair].x)));
            block_amax = fmaxf(block_amax, fabsf(__bfloat162float(cached[pair].y)));
        }

        float coeff;
        uint8_t stored_scale;
        localcta_block_quant_params(block_amax, chunk_s_enc, chunk_sg, coeff, stored_scale);

        uint64_t packed = 0;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const bf16_2 v = cached[pair];
            packed |= static_cast<uint64_t>(quantize_fp4_pair(
                __bfloat162float(v.x), __bfloat162float(v.y), coeff)) << (pair * 8);
        }

        const int global_row_pair = warp_row_base / 2 + row_group * 8;
        *reinterpret_cast<uint64_t*>(
            &g.col_fp4[logical_col * col_fp4_stride + global_row_pair]) = packed;

        const int global_row = warp_row_base + row_group * 16;
        const int row64 = global_row / 64;
        const int scale_in64 = (global_row % 64) / 16;
        const int scale_base = (col_chunk * col_ntk_total + row64) * 512 +
                               j_in_tile * 16 + grp * 4 + scale_in64;
        g.col_sc[scale_base] = stored_scale;
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
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
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
                for (int i = 0; i < num_red_blocks; ++i) {
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
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if constexpr (C::B_SC_SIZE == 2) tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {col_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    else if (cta_id == 0)            tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            uint32_t output_phasebits = 0xFFFF0000;
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                (void)row_block_idx;
                wait(outputs_finished, get_phasebit<1>(output_phasebits, 0));
                tensor_after_thread_sync();

                for (int i = 0; i < num_red_blocks; i++) {
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16);
                        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0])+16*32*ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                        if constexpr (C::B_SC_SIZE == 2) {
                            auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16+16);
                            auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0])+16*32*ii);
                            load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                        }
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) {
                        mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                inputs_finished[stage]);
                    } else {
                        mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                 A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                 B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
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
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        static_assert(SUBTILE_COLS == 32, "localCTA SiLU dgrad producer assumes 32-column epilogue slices");
        __shared__ bf16_2 staged0[WARPGROUP_WARPS][4][16][33];
        __shared__ bf16_2 staged1[WARPGROUP_WARPS][4][16][33];
        __shared__ float warp_amax0[2][WARPGROUP_WARPS];
        __shared__ float warp_amax1[2][WARPGROUP_WARPS];
        __shared__ float chunk_s_enc0[2];
        __shared__ float chunk_s_enc1[2];
        __shared__ float chunk_sg0[2];
        __shared__ float chunk_sg1[2];
        uint32_t output_phasebits = 0xFFFF0000;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            const int wg_warp = warpgroup::warpid();
            const int lane_id = warp::laneid();
            const int warp_row_base = (row_block_idx * 2 + cta_id) * (C::Mb / 2) + wg_warp * 32;

            subtile_rt D_acc[C::EPI_PIPE_DEPTH];
            wait(outputs_arrived, get_phasebit<0>(output_phasebits, 0));
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                warpgroup::load_async(
                    D_acc[epi],
                    out_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            const float gs = g.A_sg[row_block_idx * g.A_sg_stride] *
                             g.B_sg[col_block_idx * g.B_sg_stride];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                warp::mul(D_acc[epi], D_acc[epi], gs);
            }
            warpgroup::sync(1);
            tensor_after_thread_sync();
            update_phasebit<0>(output_phasebits, 0);

            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                float local_amax0 = 0.0f;
                float local_amax1 = 0.0f;
                bf16_2 (*pairs0)[16][33] = staged0[wg_warp];
                bf16_2 (*pairs1)[16][33] = staged1[wg_warp];
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int epi = half * 4 + e;
                    const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    stage_silu_deriv_pairs<C>(
                        g, D_acc[epi], pairs0, pairs1, e, lane_id, warp_row_base,
                        col_start, local_amax0, local_amax1);
                }
                local_amax0 = warp_reduce_max(local_amax0);
                local_amax1 = warp_reduce_max(local_amax1);
                if (lane_id == 0) {
                    warp_amax0[half][wg_warp] = local_amax0;
                    warp_amax1[half][wg_warp] = local_amax1;
                }
                warpgroup::sync(1);
                if (wg_warp == 0 && lane_id == 0) {
                    float amax0 = 0.0f;
                    float amax1 = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < WARPGROUP_WARPS; ++w) {
                        amax0 = fmaxf(amax0, warp_amax0[half][w]);
                        amax1 = fmaxf(amax1, warp_amax1[half][w]);
                    }
                    chunk_s_enc0[half] = localcta_encode_scale(amax0);
                    chunk_s_enc1[half] = localcta_encode_scale(amax1);
                    chunk_sg0[half] = amax0 / LOCALCTA_GLOBAL_SCALE_NUM;
                    chunk_sg1[half] = amax1 / LOCALCTA_GLOBAL_SCALE_NUM;

                    const int row_chunk = row_block_idx * 2 + cta_id;
                    const int col_chunk = col_block_idx * 2 + half;
                    const int row_sg_cols = (2 * g.H) / 128;
                    const int split_chunk_offset = g.H / 128;
                    const int col_sg_cols = g.M / 128;
                    g.row_sg[row_chunk * row_sg_cols + col_chunk] = chunk_sg0[half];
                    g.row_sg[row_chunk * row_sg_cols + split_chunk_offset + col_chunk] = chunk_sg1[half];
                    g.col_sg[col_chunk * col_sg_cols + row_chunk] = chunk_sg0[half];
                    g.col_sg[(split_chunk_offset + col_chunk) * col_sg_cols + row_chunk] = chunk_sg1[half];
                }
                warpgroup::sync(1);
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int epi = half * 4 + e;
                    const int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    quantize_rows_from_stage<C>(
                        g, pairs0, e, lane_id, warp_row_base, col_start,
                        chunk_s_enc0[half], chunk_sg0[half]);
                    quantize_cols_from_stage<C>(
                        g, pairs0, e, lane_id, warp_row_base, col_start,
                        chunk_s_enc0[half], chunk_sg0[half]);
                    quantize_rows_from_stage<C>(
                        g, pairs1, e, lane_id, warp_row_base, g.H + col_start,
                        chunk_s_enc1[half], chunk_sg1[half]);
                    quantize_cols_from_stage<C>(
                        g, pairs1, e, lane_id, warp_row_base, g.H + col_start,
                        chunk_s_enc1[half], chunk_sg1[half]);
                }
                warpgroup::sync(1);
            }
        }
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

} // namespace nvfp4_localcta_silu_dgrad_quant_gemm
