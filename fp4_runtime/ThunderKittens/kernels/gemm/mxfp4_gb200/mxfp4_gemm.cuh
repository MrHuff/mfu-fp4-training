#pragma once
// ================================================================
// MXFP4 Standard GEMM Kernel — Kb=256 with MMA_PER_TILE
// Single GEMM: D = A × B^T  (no global scale, E8M0 per-block scales)
// Mirrors nvfp4_gemm architecture for higher compute density.
// ================================================================

#include "kittens.cuh"
#include "mxfp4_rope_epilogue.cuh"
#include "mxfp4_launch_config.cuh"
#include "../common/h_mxfp4_tile_carrier.cuh"

using namespace kittens;

namespace mxfp4_gemm {

template <
    int _Nb,
    int _LOAD_PIPE_DEPTH,
    int _EPI_PIPE_DEPTH,
    int _SUPERGROUP_SIZE,
    int _NUM_D_TILES,
    bool _OVERLAP_EPI,
    int _Kb = 256,
    bool _ROPE_LIVE64_RHT32 = false,
    bool _FUSE_RESIDUAL = false,
    bool _OUTPUT_SCALE = false,
    bool _FUSE_H_MX_CARRIER = false,
    bool _FUSE_C1_RMS_CTA = false,
    bool _USE_PDL = mxfp4_launch::default_use_pdl
>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_Kb == 128 || _Kb == 256, "Kb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5, "LOAD_PIPE_DEPTH must be greater than 0 and at most 5");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be greater than 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2, "NUM_D_TILES must be at least 2 if EPI_PIPE_DEPTH > 1");

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = _USE_PDL;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;
    static constexpr bool ROPE_LIVE64_RHT32 = _ROPE_LIVE64_RHT32;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr bool FUSE_RESIDUAL = _FUSE_RESIDUAL;
    static constexpr bool OUTPUT_SCALE = _OUTPUT_SCALE;
    static constexpr bool FUSE_H_MX_CARRIER = _FUSE_H_MX_CARRIER;
    static constexpr bool FUSE_C1_RMS_CTA = _FUSE_C1_RMS_CTA;
    static_assert(
        !FUSE_H_MX_CARRIER ||
            (_FUSE_RESIDUAL && !_OVERLAP_EPI && _Nb == 256),
        "H MX carrier requires residual, retained epilogue, and Nb=256");
    static_assert(
        !FUSE_C1_RMS_CTA ||
            (_FUSE_RESIDUAL && !_OVERLAP_EPI && _Nb == 256),
        "exact row RMS carrier requires residual, retained epilogue, and Nb=256");
    static_assert(!(FUSE_H_MX_CARRIER && FUSE_C1_RMS_CTA),
                  "exact row RMS and tile RMS carriers are distinct epilogues");
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/128;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
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
    // FP4 tiles: each element is 4 bits, stored as fp4e2m1_2 (packed pairs)
    // With Kb=256: A tile is 128×128 packed (256×256 FP4 elements)
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;  // E8M0 scale, block-32, covers 128 K
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;  // E8M0 scale, block-32, covers 128 K
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    // Scale global: (M/128) x (K/128) x 32 x 16 — each tile covers 128 K elements
    using A_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, B_sc_tile>;
    using D_gl       = gl<bf16,      1,  1, -1, -1, D_tile>;

    A_fp4x2_gl A;       // M x (K/2)
    A_sc_gl    A_sc;    // (M/128) x (K/128) x 32 x 16
    B_fp4x2_gl B;       // N x (K/2)
    B_sc_gl    B_sc;    // (N/128) x (K/128) x 32 x 16
    D_gl       D;       // M x N
    CUtensorMap R_tma;   // optional residual descriptor, M x N
    mxfp4_rope_epilogue::rope_desc rope;
    mxfp4_rope_epilogue::rope_live64_desc rope_live64;
    const float* output_scale;     // optional scalar epilogue multiplier
    float* row_rms_partial;        // optional [M,N/256] BF16-output row sumsq
    int row_rms_partial_stride;
    uint8_t* h_row_fp4;            // H row payload [M,N/2]
    uint8_t* h_row_sc;             // H row E8M0 [M/128,N/128,32,16]
    uint8_t* h_col_fp4;            // H column payload [N,M/2]
    uint8_t* h_col_sc;             // H column E8M0 [N/128,M/128,32,16]
    float* h_r_tile;               // H inverse RMS [M/128,N/128]
    const bf16* h_gamma;           // H post-RMS gamma [N]
    int h_rows;
    int h_cols;
    float h_eps;
    const uint8_t* tilemask_ptr;   // optional [mask_rows, mask_cols] activity mask
    int            tilemask_rows;
    int            tilemask_cols;
    bool           tilemask_transposed;
    bool           output_causal = false;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    // Scale struct: MMA_PER_TILE (=2) scale tiles per pipeline stage
    // to cover the full Kb=256 range (2 × 128-wide E8M0 tiles)
    struct input_scales_t {
        A_sc_tile A[C::MMA_PER_TILE];
        B_sc_tile B[C::B_SC_SIZE * C::MMA_PER_TILE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        return dim3(min((D.rows()/(C::Mb/2))*(D.cols()/C::Nb), num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

template <typename C>
__device__ inline bool reduction_iter_active(
    const globals<C> &g,
    int row_tile_128,
    int red_iter
) {
    if (g.tilemask_ptr == nullptr) {
        return true;
    }

    constexpr int RED_TILES_PER_ITER = C::Kb / 128;
    const int red_tile_base = red_iter * RED_TILES_PER_ITER;

    auto tile_active = [&](int red_tile_128) {
        if (g.tilemask_transposed) {
            if (red_tile_128 >= g.tilemask_rows || row_tile_128 >= g.tilemask_cols) {
                return false;
            }
            return g.tilemask_ptr[red_tile_128 * g.tilemask_cols + row_tile_128] != 0;
        }
        if (row_tile_128 >= g.tilemask_rows || red_tile_128 >= g.tilemask_cols) {
            return false;
        }
        return g.tilemask_ptr[row_tile_128 * g.tilemask_cols + red_tile_128] != 0;
    };

    bool active = false;
    #pragma unroll
    for (int red_tile_offset = 0; red_tile_offset < RED_TILES_PER_ITER; ++red_tile_offset) {
        active = active || tile_active(red_tile_base + red_tile_offset);
    }
    return active;
}

template <typename C>
__device__ inline bool output_block_active(
    const globals<C> &g,
    int row_block_idx,
    int col_block_idx
) {
    if (!g.output_causal) {
        return true;
    }
    // A spatial block covers 256 output rows and C::Nb output columns.  The
    // dP path only consumes lower-triangular 128x128 score tiles, so any
    // 256-column block strictly above the 256-row block can be skipped.
    return col_block_idx <= row_block_idx;
}

template <typename C>
__device__ inline void maybe_add_residual_tile(
    const globals<C> &g,
    typename globals<C>::D_tile &smem_tile,
    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> &D_reg,
    semaphore &residual_load_arrived,
    uint32_t &residual_load_phase,
    int row_tile,
    int col_tile
) {
    if constexpr (!C::FUSE_RESIDUAL) {
        return;
    }

    else {
        tma_dev_proxy<typename globals<C>::D_gl> R_proxy(&g.R_tma);
        if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
            tma::expect_bytes(residual_load_arrived, sizeof(typename globals<C>::D_tile));
            tma::load_async(smem_tile, R_proxy, {row_tile, col_tile}, residual_load_arrived);
        }
        wait(residual_load_arrived, residual_load_phase);
        residual_load_phase ^= 1;
        warpgroup::sync(1);

        rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> R_reg;
        warpgroup::load(R_reg, smem_tile);
        warpgroup::sync(1);
        warp::add(D_reg, D_reg, R_reg);
    }
}

template <typename C, typename RT>
__device__ inline void accumulate_c1_cta_row_rms_partial(
    const globals<C>& g,
    const RT& D_reg,
    float* scratch,
    bool initialize,
    bool finalize,
    int row_tile,
    int col_block
) {
    if constexpr (C::FUSE_C1_RMS_CTA) {
        rt_fl<RT::rows, RT::cols> D_fl;
        rt_fl<RT::rows, RT::cols> D_sq;
        typename decltype(D_fl)::col_vec row_sums;
        warp::copy(D_fl, D_reg);
        warp::mul(D_sq, D_fl, D_fl);
        warp::row_sum(row_sums, D_sq);
        const int lane = warp::laneid();
        const int warp_row_base =
            row_tile * (C::Mb / 2) +
            warpgroup::warpid() * decltype(row_sums)::length;
        const int scratch_base =
            warpgroup::warpid() * decltype(row_sums)::length;
        if ((lane & 3) == 0) {
            #pragma unroll
            for (int i = 0; i < decltype(row_sums)::outer_dim; ++i) {
                const int row_x = warp_row_base + i * 16 + lane / 4;
                const int row_y = row_x + 8;
                const int scratch_x = scratch_base + i * 16 + lane / 4;
                const int scratch_y = scratch_x + 8;
                const float sum_x = row_sums[i][0].x;
                const float sum_y = row_sums[i][0].y;
                if (finalize) {
                    g.row_rms_partial[
                        row_x * g.row_rms_partial_stride + col_block] =
                        initialize ? sum_x : scratch[scratch_x] + sum_x;
                    g.row_rms_partial[
                        row_y * g.row_rms_partial_stride + col_block] =
                        initialize ? sum_y : scratch[scratch_y] + sum_y;
                } else if (initialize) {
                    scratch[scratch_x] = sum_x;
                    scratch[scratch_y] = sum_y;
                } else {
                    scratch[scratch_x] += sum_x;
                    scratch[scratch_y] += sum_y;
                }
            }
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
        if constexpr (C::FUSE_RESIDUAL) {
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.R_tma)) : "memory");
        }
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int num_col_blocks = g.D.cols() / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_iters_per_block = 2 * g.A.cols() / C::Kb;  // Half the iters vs Kb=128
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Declare tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore residual_load_arrived;
    __shared__ float h_reduce[
        C::FUSE_H_MX_CARRIER ? WARPGROUP_WARPS + 1 : 1];
    __shared__ bf16_2 h_stage
        [C::FUSE_H_MX_CARRIER ? WARPGROUP_WARPS : 1]
        [C::FUSE_H_MX_CARRIER ? C::Nb / C::EPI_PIPE_DEPTH / 2 : 1]
        [C::FUSE_H_MX_CARRIER ? 33 : 1];
    __shared__ float c1_row_sums[
        C::FUSE_C1_RMS_CTA ? C::Mb / 2 : 1];
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
        if constexpr (C::FUSE_RESIDUAL) {
            init_semaphore(residual_load_arrived, 0, 1);
        }
    }
    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait_aligned();

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input FP4 tiles to shared memory
            if constexpr (C::USE_PDL) pdl::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                    continue;
                }
                const int row_tile_128_0 = row_block_idx * 2 + 0;
                const int row_tile_128_1 = row_block_idx * 2 + 1;
                for (int i = 0; i < num_iters_per_block; ++i) {
                    const bool block_iter_active =
                        reduction_iter_active(g, row_tile_128_0, i) ||
                        reduction_iter_active(g, row_tile_128_1, i);
                    if (!block_iter_active) continue;
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            // Load input scales to shared memory
            // Each iteration loads MMA_PER_TILE (=2) scale tiles per A and B
            if constexpr (C::USE_PDL) pdl::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                    continue;
                }
                const int row_tile_128_0 = row_block_idx * 2 + 0;
                const int row_tile_128_1 = row_block_idx * 2 + 1;

                for (int i = 0; i < num_iters_per_block; ++i) {
                    const bool block_iter_active =
                        reduction_iter_active(g, row_tile_128_0, i) ||
                        reduction_iter_active(g, row_tile_128_1, i);
                    if (!block_iter_active) continue;
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    // Load MMA_PER_TILE A scale tiles (each covers 128 K elements)
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        tma::cluster::load_async(input_scales[stage].A[k], g.A_sc,
                            {row_block_idx*2 + cta_id, i*C::MMA_PER_TILE + k, 0, 0},
                            scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    }
                    // Load B scale tiles
                    if constexpr (C::B_SC_SIZE == 2) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            tma::cluster::load_async(
                                input_scales[stage].B[cta_id * C::MMA_PER_TILE + k], g.B_sc,
                                {col_block_idx*2 + cta_id, i*C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        }
                    } else if (cta_id == 0) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            tma::cluster::load_async(
                                input_scales[stage].B[k], g.B_sc,
                                {col_block_idx, i*C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        }
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // Launch tensor core matrix multiply
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            // Scale tensor memory: MMA_PER_TILE × per-stage entries
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                    continue;
                }
                const int row_tile_128_0 = row_block_idx * 2 + 0;
                const int row_tile_128_1 = row_block_idx * 2 + 1;
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                bool issued_mma = false;
                for (int i = 0; i < num_iters_per_block; i++) {
                    const bool block_iter_active =
                        reduction_iter_active(g, row_tile_128_0, i) ||
                        reduction_iter_active(g, row_tile_128_1, i);
                    if (!block_iter_active) continue;
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    // Load MMA_PER_TILE scale subtiles into tensor memory
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; k++) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*16 + k*16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*C::B_SC_SIZE*16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                        if constexpr (C::B_SC_SIZE == 2) {
                            auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*C::B_SC_SIZE*16 + 16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[C::MMA_PER_TILE + k]);
                        }
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
                update_phasebit<1>(phasebits, 0);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // Consumer group — no global scale needed for MXFP4
        // The preceding clustered GEMM releases tensor memory before its PDL
        // arrival. Do not provision the dependent cluster until then.
        if constexpr (C::USE_PDL) warpgroup::pdl::wait();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        uint32_t residual_load_phase = 0;
        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                continue;
            }
            const int row_tile_128 = row_block_idx * 2 + cta_id;
            bool block_has_active = false;
            #pragma unroll
            for (int i = 0; i < num_iters_per_block; ++i) {
                block_has_active = block_has_active ||
                    reduction_iter_active(g, row_tile_128, i);
            }

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Load the output from tensor memory into registers and store to HBM
            // Apply MXFP4 alpha scaling (1/36) in-register before store.
            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
            float gemm_scale = MXFP4_ALPHA;
            if constexpr (C::OUTPUT_SCALE) {
                gemm_scale *= *g.output_scale;
            }
            if (!block_has_active) {
                if constexpr (C::OVERLAP_EPI) {
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        const int smem_slot = i % C::NUM_D_TILES;
                        rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg {};
                        if (i == C::EPI_PIPE_DEPTH - 1) {
                            warpgroup::sync(1);
                            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                        }
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        maybe_add_residual_tile<C>(
                            g, output_tiles.D[smem_slot], D_reg,
                            residual_load_arrived, residual_load_phase,
                            row_block_idx * 2 + cta_id,
                            col_block_idx * C::EPI_PIPE_DEPTH + i);
                        warpgroup::store(output_tiles.D[smem_slot], D_reg);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[smem_slot], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                    }
                } else {
                    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH] {};
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        const int smem_slot = i % C::NUM_D_TILES;
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        maybe_add_residual_tile<C>(
                            g, output_tiles.D[smem_slot], D_reg[i],
                            residual_load_arrived, residual_load_phase,
                            row_block_idx * 2 + cta_id,
                            col_block_idx * C::EPI_PIPE_DEPTH + i);
                        warpgroup::store(output_tiles.D[smem_slot], D_reg[i]);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[smem_slot], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                    }
                }
            } else if constexpr (C::FUSE_H_MX_CARRIER) {
                constexpr int SLICE = C::Nb / C::EPI_PIPE_DEPTH;
                constexpr int SLICES_PER_TILE = 128 / SLICE;
                static_assert(
                    (SLICE == 32 && C::EPI_PIPE_DEPTH == 8) ||
                    (SLICE == 64 && C::EPI_PIPE_DEPTH == 4));
                using H_rt_fl = rt_fl<C::Mb / 8, SLICE>;
                using H_rt_bf = rt_bf<C::Mb / 8, SLICE>;
                const int row_tile = row_block_idx * 2 + cta_id;
                const int warp_row =
                    row_tile * 128 + warpgroup::warpid() * 32;
                #pragma unroll
                for (int tile = 0; tile < 2; ++tile) {
                    H_rt_bf z[SLICES_PER_TILE];
                    #pragma unroll
                    for (int q = 0; q < SLICES_PER_TILE; ++q) {
                        const int i = tile * SLICES_PER_TILE + q;
                        H_rt_fl f;
                        warpgroup::load_async(
                            f,
                            out_tm.template subtile<full_tt_fl<SLICE>>(
                                0, SLICE * i));
                        warp::mul(f, f, gemm_scale);
                        warp::copy(z[q], f);
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    if (tile == 1) {
                        warpgroup::tma::cluster::arrive(
                            outputs_finished, 0, 1);
                    }

                    #pragma unroll
                    for (int q = 0; q < SLICES_PER_TILE; ++q) {
                        const int i = tile * SLICES_PER_TILE + q;
                        const int slot = i % C::NUM_D_TILES;
                        warpgroup::tma::store_async_read_wait<
                            C::NUM_D_TILES - 1>();
                        warpgroup::sync(1);
                        maybe_add_residual_tile<C>(
                            g,
                            output_tiles.D[slot],
                            z[q],
                            residual_load_arrived,
                            residual_load_phase,
                            row_tile,
                            col_block_idx * C::EPI_PIPE_DEPTH + i);
                        warpgroup::store(output_tiles.D[slot], z[q]);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<
                            dim::ROW, cache_policy::EVICT_FIRST>(
                            g.D,
                            output_tiles.D[slot],
                            {row_tile,
                             col_block_idx * C::EPI_PIPE_DEPTH + i});
                    }

                    float s = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < SLICES_PER_TILE; ++q) {
                        s += h_mxfp4_tile_carrier::sumsq(z[q]);
                    }
                    const float r = h_mxfp4_tile_carrier::tile_rsqrt(
                        s, h_reduce, g.h_eps);
                    if (threadIdx.x % 128 == 0 &&
                        warpgroup::warpid() == 0) {
                        g.h_r_tile[
                            row_tile * (g.h_cols / 128) +
                            col_block_idx * 2 + tile] = r;
                    }
                    #pragma unroll
                    for (int q = 0; q < SLICES_PER_TILE; ++q) {
                        const int i = tile * SLICES_PER_TILE + q;
                        const int col =
                            col_block_idx * C::Nb + i * SLICE;
                        auto* pairs = h_stage[warpgroup::warpid()];
                        h_mxfp4_tile_carrier::normalize_gamma_to_stage(
                            z[q], pairs, r, g.h_gamma, col);
                        __syncwarp();
                        #pragma unroll
                        for (int sub = 0; sub < SLICE / 32; ++sub) {
                            h_mxfp4_tile_carrier::emit_32x32(
                                g, pairs + sub * 16, warp_row,
                                col + sub * 32);
                        }
                    }
                }
            } else if constexpr (C::OVERLAP_EPI) {
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    const int smem_slot = i % C::NUM_D_TILES;
                    rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    }
                    warp::mul(D_reg_fl, D_reg_fl, gemm_scale);
                    if (g.rope_live64.enabled()) {
                        if constexpr (C::ROPE_LIVE64_RHT32) {
                            mxfp4_rope_epilogue::apply_inplace_live64_rht32(
                                D_reg_fl,
                                g.rope_live64,
                                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                                (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                            );
                        } else {
                            mxfp4_rope_epilogue::apply_inplace_live64(
                                D_reg_fl,
                                g.rope_live64,
                                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                                (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                            );
                        }
                    } else {
                        mxfp4_rope_epilogue::apply_inplace(
                            D_reg_fl,
                            g.rope,
                            (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                            (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                        );
                    }
                    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg;
                    warp::copy(D_reg, D_reg_fl);
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    maybe_add_residual_tile<C>(
                        g, output_tiles.D[smem_slot], D_reg,
                        residual_load_arrived, residual_load_phase,
                        row_block_idx * 2 + cta_id,
                        col_block_idx * C::EPI_PIPE_DEPTH + i);
                    warpgroup::store(output_tiles.D[smem_slot], D_reg);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[smem_slot], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                }
            } else {
                rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                    warp::mul(D_reg_fl, D_reg_fl, gemm_scale);
                    if (g.rope_live64.enabled()) {
                        if constexpr (C::ROPE_LIVE64_RHT32) {
                            mxfp4_rope_epilogue::apply_inplace_live64_rht32(
                                D_reg_fl,
                                g.rope_live64,
                                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                                (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                            );
                        } else {
                            mxfp4_rope_epilogue::apply_inplace_live64(
                                D_reg_fl,
                                g.rope_live64,
                                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                                (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                            );
                        }
                    } else {
                        mxfp4_rope_epilogue::apply_inplace(
                            D_reg_fl,
                            g.rope,
                            (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                            (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                        );
                    }
                    warp::copy(D_reg[i], D_reg_fl);
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    const int smem_slot = i % C::NUM_D_TILES;
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    maybe_add_residual_tile<C>(
                        g, output_tiles.D[smem_slot], D_reg[i],
                        residual_load_arrived, residual_load_phase,
                        row_block_idx * 2 + cta_id,
                        col_block_idx * C::EPI_PIPE_DEPTH + i);
                    warpgroup::store(output_tiles.D[smem_slot], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D, output_tiles.D[smem_slot], {row_block_idx * 2 + cta_id, col_block_idx * C::EPI_PIPE_DEPTH + i});
                    if constexpr (C::FUSE_C1_RMS_CTA) {
                        accumulate_c1_cta_row_rms_partial<C>(
                            g, D_reg[i], c1_row_sums,
                            i == 0, i == C::EPI_PIPE_DEPTH - 1,
                            row_block_idx * 2 + cta_id,
                            col_block_idx);
                    }
                }
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        // Ensure all TMA stores have committed before signaling the next
        // kernel can start (matches NVFP4 pattern).
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
    }

    // Both CTAs use peer DSM/TMEM traffic. Keep either CTA alive until its
    // peer has retired all tail operations targeting the cluster.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

} // namespace mxfp4_gemm
