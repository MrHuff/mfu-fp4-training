#pragma once
// ================================================================
// MXFP4 Fused Cross-Entropy Kernel v2 — Ping-Pong MMA
//
// Architecture: Ping-pong MMA with TWO TMEM accumulators.
// While consumer reads accum[A], MMA writes accum[B], and vice versa.
// This eliminates the TMEM drain stall between output blocks.
//
// Constraint: Nb=128 (two 128-wide accumulators fit in 512-col TMEM).
// ================================================================

#include "kittens.cuh"
#include "mxfp4_launch_config.cuh"

using namespace kittens;

namespace mxfp4_cce_v2 {

// Atomic logaddexp: atomically merge val into *addr
__device__ inline void atomic_logaddexp(float* addr, float val) {
    unsigned int* addr_as_uint = (unsigned int*)addr;
    unsigned int old = *addr_as_uint;
    while (true) {
        float old_f = __uint_as_float(old);
        float mx = fmaxf(old_f, val);
        float new_f;
        if (__isinff(mx) && mx < 0.f) {
            new_f = -INFINITY;
        } else {
            new_f = mx + __logf(__expf(old_f - mx) + __expf(val - mx));
        }
        unsigned int assumed = old;
        old = atomicCAS(addr_as_uint, assumed, __float_as_uint(new_f));
        if (old == assumed) break;
    }
}

// Fixed Nb=128 for ping-pong (2 × 128 = 256 TMEM cols for accumulators)
template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = mxfp4_launch::default_use_pdl;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = 4;  // 128 / 32 = 4 subtiles
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;  // Fixed for ping-pong
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = 1;  // Nb/128 = 1
    static constexpr int MMA_PER_TILE = 2;  // Kb/128

    static constexpr int NUM_D_TILES = 2;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, B_sc_tile>;
    using D_gl       = gl<bf16,      1,  1, -1, -1, D_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;
    D_gl       D_scratch;

    float* lse;
    float* neg_correct_logit;
    const int64_t* targets;
    int M;
    int N;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A[C::MMA_PER_TILE];
        B_sc_tile B[C::MMA_PER_TILE];  // B_SC_SIZE=1
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        int padded_M = A.rows();
        int padded_N = B.rows();
        int num_row_blocks = padded_M / (C::Mb / 2);
        int num_col_blocks = padded_N / C::Nb;
        int total = num_row_blocks * num_col_blocks;
        int max_blocks = num_sms();
        int grid_size = min(total, max_blocks);
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(grid_size);
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

// ======================== CONSUMER EPILOGUE ========================
// Processes logsumexp from registers after batch-loading from TMEM.
template <typename C, typename G>
__device__ inline void consumer_epilogue(
    const globals<C> &g,
    int tile_row_base, int warp_row_base, int col_block_idx,
    int lane_id,
    auto& D_regs  // subtile_rt[EPI_PIPE_DEPTH]
) {
    constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
    constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;

    using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
    using col_vec_t = typename subtile_rt::col_vec;

    // Precompute per-lane target info  
    int my_targets[subtile_rt::height];
    #pragma unroll
    for (int i = 0; i < subtile_rt::height; i++) {
        int global_row_x = warp_row_base + i * 16 + lane_id / 4;
        my_targets[i] = (global_row_x < g.M) ? (int)g.targets[global_row_x] : -1;
    }
    int my_targets_y[subtile_rt::height];
    #pragma unroll
    for (int i = 0; i < subtile_rt::height; i++) {
        int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
        my_targets_y[i] = (global_row_y < g.M) ? (int)g.targets[global_row_y] : -1;
    }

    // Online logsumexp running state
    col_vec_t running_max;
    col_vec_t running_sumexp;
    #pragma unroll
    for (int i = 0; i < running_max.outer_dim; i++) {
        #pragma unroll
        for (int j = 0; j < running_max.inner_dim; j++) {
            running_max[i][j].x = -INFINITY;
            running_max[i][j].y = -INFINITY;
            running_sumexp[i][j].x = 0.f;
            running_sumexp[i][j].y = 0.f;
        }
    }

    // Process ALL subtiles from registers
    #pragma unroll
    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
        warp::mul(D_regs[epi], D_regs[epi], MXFP4_ALPHA);

        int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
        #pragma unroll
        for (int i = 0; i < subtile_rt::height; i++) {
            int tgt_x = my_targets[i];
            if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
                int local_col = tgt_x - col_start;
                int j_idx = local_col / 16;           // TILE_COL_DIM<float>=16
                int within_tile = local_col % 16;
                int k_half = within_tile / 8;          // col half: 0=left(0-7), 1=right(8-15)
                int pair_pos = (within_tile % 8) / 2;  // which lane pair (0-3)
                if ((lane_id % 4) == pair_pos) {
                    int k_idx = k_half * 2;
                    float val;
                    if ((local_col & 1) == 0)
                        val = D_regs[epi].tiles[i][j_idx].data[k_idx].x;
                    else
                        val = D_regs[epi].tiles[i][j_idx].data[k_idx].y;
                    int global_row = warp_row_base + i * 16 + lane_id / 4;
                    atomicAdd(&g.neg_correct_logit[global_row], -val);
                }
            }
            int tgt_y = my_targets_y[i];
            if (tgt_y >= col_start && tgt_y < col_start + SUBTILE_COLS) {
                int local_col = tgt_y - col_start;
                int j_idx = local_col / 16;
                int within_tile = local_col % 16;
                int k_half = within_tile / 8;
                int pair_pos = (within_tile % 8) / 2;
                if ((lane_id % 4) == pair_pos) {
                    int k_idx = k_half * 2 + 1;
                    float val;
                    if ((local_col & 1) == 0)
                        val = D_regs[epi].tiles[i][j_idx].data[k_idx].x;
                    else
                        val = D_regs[epi].tiles[i][j_idx].data[k_idx].y;
                    int global_row = warp_row_base + i * 16 + 8 + lane_id / 4;
                    atomicAdd(&g.neg_correct_logit[global_row], -val);
                }
            }
        }

        col_vec_t tile_max;
        warp::row_max(tile_max, D_regs[epi]);
        warp::sub_row(D_regs[epi], D_regs[epi], tile_max);
        warp::exp(D_regs[epi], D_regs[epi]);
        col_vec_t tile_sumexp;
        warp::row_sum(tile_sumexp, D_regs[epi]);
        #pragma unroll
        for (int i_v = 0; i_v < col_vec_t::outer_dim; i_v++) {
            #pragma unroll
            for (int j_v = 0; j_v < col_vec_t::inner_dim; j_v++) {
                float old_max_x = running_max[i_v][j_v].x;
                float new_max_x = fmaxf(old_max_x, tile_max[i_v][j_v].x);
                float old_max_y = running_max[i_v][j_v].y;
                float new_max_y = fmaxf(old_max_y, tile_max[i_v][j_v].y);
                running_sumexp[i_v][j_v].x = running_sumexp[i_v][j_v].x * __expf(old_max_x - new_max_x) +
                                             tile_sumexp[i_v][j_v].x * __expf(tile_max[i_v][j_v].x - new_max_x);
                running_sumexp[i_v][j_v].y = running_sumexp[i_v][j_v].y * __expf(old_max_y - new_max_y) +
                                             tile_sumexp[i_v][j_v].y * __expf(tile_max[i_v][j_v].y - new_max_y);
                running_max[i_v][j_v].x = new_max_x;
                running_max[i_v][j_v].y = new_max_y;
            }
        }
    }

    // Finalize LSE
    const int warpid_in_wg = warpgroup::warpid();
    int warp_row_offset = warpid_in_wg * (C::Mb / 8);
    
    if (lane_id % 4 == 0) {
        #pragma unroll
        for (int i = 0; i < col_vec_t::outer_dim; i++) {
            int global_row_x = tile_row_base + warp_row_offset + i * 16 + lane_id / 4;
            int global_row_y = global_row_x + 8;
            
            float max_x = running_max[i][0].x;
            float max_y = running_max[i][0].y;
            float sumexp_x = running_sumexp[i][0].x;
            float sumexp_y = running_sumexp[i][0].y;

            if (global_row_x < g.M && max_x > -INFINITY) {
                float local_lse = max_x + __logf(sumexp_x);
                atomic_logaddexp(&g.lse[global_row_x], local_lse);
            }
            if (global_row_y < g.M && max_y > -INFINITY) {
                float local_lse = max_y + __logf(sumexp_y);
                atomic_logaddexp(&g.lse[global_row_y], local_lse);
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
        g.D_scratch.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;

    const int padded_M = g.A.rows();
    const int padded_N = g.B.rows();
    const int num_row_blocks = padded_M / C::Mb;
    const int num_col_blocks = padded_N / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_iters_per_block = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];

    // Non-pingpong: single output semaphores (identical to v1)
    // Pingpong: two sets
    static constexpr int NUM_PHASES = C::PINGPONG ? 2 : 1;
    __shared__ semaphore outputs_arrived[NUM_PHASES];
    __shared__ semaphore outputs_finished[NUM_PHASES];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        for (int i = 0; i < NUM_PHASES; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::arrive_aligned();

    // ======================== PRODUCER ========================
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
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
            everyone::tma::cluster::wait();
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
                    if (cta_id == 0) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            tma::cluster::load_async(input_scales[stage].B[k], g.B_sc,
                                {col_block_idx, i*C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        }
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // ======================== MMA WARP ========================
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            if constexpr (!C::PINGPONG) {
                // === NON-PINGPONG: single accumulator, identical to v1 ===
                auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    wait(outputs_finished[0], get_phasebit<1>(phasebits, 0));
                    tensor_after_thread_sync();
                    for (int i = 0; i < num_iters_per_block; i++) {
                        tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*16 + k*16);
                            load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                            auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                        }
                        tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                            A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                                            B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                                            inputs_finished[stage]);
                        else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                            A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                                            B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                                            inputs_finished[stage]);
                        update_phasebit<0>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                    tensor_commit<2>(outputs_arrived[0]);
                    update_phasebit<1>(phasebits, 0);
                }
            } else {
                // === PINGPONG: Two accumulators at TMEM offsets 0 and 128 ===
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

                int phase = 0;

                auto do_mma_block = [&](auto& accum) {
                    for (int i = 0; i < num_iters_per_block; i++) {
                        tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; k++) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*16 + k*16);
                            load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                            auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage*C::MMA_PER_TILE*32 + k*16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                        }
                        tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        if (i == 0) mm2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                            A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                                            B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                                            inputs_finished[stage]);
                        else       mma2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                            A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*16>>(stage * C::MMA_PER_TILE * 16),
                                            B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE*32>>(stage * C::MMA_PER_TILE * 32),
                                            inputs_finished[stage]);
                        update_phasebit<0>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                };

                // Single-barrier protocol (same as v1) but alternate accumulator
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    wait(outputs_finished[0], get_phasebit<1>(phasebits, 0));
                    tensor_after_thread_sync();

                    if (phase == 0) do_mma_block(out_tm_0);
                    else            do_mma_block(out_tm_1);

                    tensor_commit<2>(outputs_arrived[0]);
                    update_phasebit<1>(phasebits, 0);
                    phase ^= 1;
                }
            }
        }
    // ======================== CONSUMER ========================
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if constexpr (C::USE_PDL) warpgroup::pdl::wait();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;

        const int lane_id = threadIdx.x % 32;

        if constexpr (!C::PINGPONG) {
            // === NON-PINGPONG: identical to v1 ===
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                wait(outputs_arrived[0], get_phasebit<0>(phasebits, 0));

                int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

                subtile_rt D_regs[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    warpgroup::load_async(D_regs[epi], out_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished[0], 0, 1);

                consumer_epilogue<C, G>(g, tile_row_base, warp_row_base, col_block_idx, lane_id, D_regs);

                update_phasebit<0>(phasebits, 0);
            }
        } else {
            // === PINGPONG: single barrier, alternating accumulator ===
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            int phase = 0;

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                // Single barrier — same as v1
                wait(outputs_arrived[0], get_phasebit<0>(phasebits, 0));

                int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

                subtile_rt D_regs[C::EPI_PIPE_DEPTH];
                auto load_from_accum = [&](auto& accum) {
                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                        warpgroup::load_async(D_regs[epi], accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                    }
                };
                if (phase == 0) load_from_accum(out_tm_0);
                else            load_from_accum(out_tm_1);
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                warpgroup::tma::cluster::arrive(outputs_finished[0], 0, 1);

                consumer_epilogue<C, G>(g, tile_row_base, warp_row_base, col_block_idx, lane_id, D_regs);

                update_phasebit<0>(phasebits, 0);
                phase ^= 1;
            }
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
    }
}

} // namespace mxfp4_cce_v2
