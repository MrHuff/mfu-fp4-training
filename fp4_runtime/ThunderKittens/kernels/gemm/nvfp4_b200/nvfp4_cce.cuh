#pragma once
// ================================================================
// NVFP4 Fused Cross-Entropy Kernel — CCE variant of nvfp4_gemm
//
// Instead of materializing [M, V] logits, computes:
//   lse[M]               = logsumexp over all vocab columns
//   neg_correct_logit[M]  = -logit at target column
//
// Loss = mean(neg_correct_logit + lse)
//
// Architecture: same as nvfp4_gemm but the consumer epilogue
// computes online logsumexp via TK's row_max/row_sum + exp/sub_row,
// then atomically merges per-tile LSE into global lse[] via logaddexp.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_cce {

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

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI, int _Mb = 256, bool _USE_PDL = true, int _CLUSTER_SIZE = 2>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_Mb == 256 || _Mb == 512, "Mb must be 256 or 512");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_EPI_PIPE_DEPTH > 0);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_NUM_D_TILES > 0);

    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    static constexpr bool USE_PDL = _USE_PDL;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename C>
struct globals {
    // NVFP4 tile types (same as nvfp4_gemm)
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;
    D_gl           D_scratch;  // Tiny scratch buffer for TMA pipeline pacing

    // CCE outputs (raw pointers, not TMA)
    float* lse;                // [M] — initialized to -inf
    float* neg_correct_logit;  // [M] — initialized to 0
    const int64_t* targets;    // [M] — target indices
    int M;                     // actual M (unpadded)
    int N;                     // actual N (unpadded)

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
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
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
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

    // ======================== PRODUCER ========================
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles to shared memory
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
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
            // Load input scales to shared memory
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
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
            // Launch tensor core matrix multiplies
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
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
                    if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
            }
        }
    // ======================== CONSUMER — DEFERRED CCE EPILOGUE ========================
    //
    // Key optimization: Adapts OVERLAP_EPI=false pattern to decouple CCE from pipeline.
    //   Phase 1: Batch ALL TMEM → bf16 registers (fast). Signal outputs_finished.
    //   Phase 2: CCE + TMA store interleaved (overlaps with producer MMA).
    //
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        // NVFP4 global scale
        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;

        // Register tile and col vector types for this warp
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using bf16_subtile_rt = rt_bf<C::Mb / 8, SUBTILE_COLS>;
        using col_vec_t = typename subtile_rt::col_vec;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Online logsumexp running state per row in this warp
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

            constexpr int HALF = C::EPI_PIPE_DEPTH / 2;

            // ============ PHASE 1a: Process first HALF subtiles inline ============
            #pragma unroll
            for (int epi = 0; epi < HALF; epi++) {
                subtile_rt D_reg_fl;
                warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                warp::mul(D_reg_fl, D_reg_fl, global_scale);

                // Target extraction — skip if no target can land in this subtile
                {
                    int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    int col_end = col_start + SUBTILE_COLS;
                    if (col_start < g.N && col_end > 0) {
                        int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                        int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
                        warp::apply(D_reg_fl, D_reg_fl, [&](int local_row, int local_col, float val) -> float {
                            int global_row = warp_row_base + local_row;
                            int global_col = col_start + local_col;
                            if (global_row < g.M && global_col < g.N) {
                                int64_t target = g.targets[global_row];
                                if ((int)target == global_col) {
                                    atomicAdd(&g.neg_correct_logit[global_row], -val);
                                }
                            }
                            return val;
                        });
                    }
                }

                // Online logsumexp — specialize first subtile (running state is -inf)
                col_vec_t tile_max;
                warp::row_max(tile_max, D_reg_fl);
                warp::sub_row(D_reg_fl, D_reg_fl, tile_max);
                warp::exp(D_reg_fl, D_reg_fl);
                col_vec_t tile_sumexp;
                warp::row_sum(tile_sumexp, D_reg_fl);
                if (epi == 0) {
                    // First subtile: just copy (running_max was -inf, running_sumexp was 0)
                    #pragma unroll
                    for (int i = 0; i < col_vec_t::outer_dim; i++) {
                        #pragma unroll
                        for (int j = 0; j < col_vec_t::inner_dim; j++) {
                            running_max[i][j].x = tile_max[i][j].x;
                            running_max[i][j].y = tile_max[i][j].y;
                            running_sumexp[i][j].x = tile_sumexp[i][j].x;
                            running_sumexp[i][j].y = tile_sumexp[i][j].y;
                        }
                    }
                } else {
                    // Subsequent subtiles: merge without -INFINITY guard
                    #pragma unroll
                    for (int i = 0; i < col_vec_t::outer_dim; i++) {
                        #pragma unroll
                        for (int j = 0; j < col_vec_t::inner_dim; j++) {
                            float old_max_x = running_max[i][j].x;
                            float new_max_x = fmaxf(old_max_x, tile_max[i][j].x);
                            float old_max_y = running_max[i][j].y;
                            float new_max_y = fmaxf(old_max_y, tile_max[i][j].y);
                            running_sumexp[i][j].x = running_sumexp[i][j].x * __expf(old_max_x - new_max_x) +
                                                     tile_sumexp[i][j].x * __expf(tile_max[i][j].x - new_max_x);
                            running_sumexp[i][j].y = running_sumexp[i][j].y * __expf(old_max_y - new_max_y) +
                                                     tile_sumexp[i][j].y * __expf(tile_max[i][j].y - new_max_y);
                            running_max[i][j].x = new_max_x;
                            running_max[i][j].y = new_max_y;
                        }
                    }
                }

                // TMA store for pacing
                bf16_subtile_rt D_reg_bf_tmp;
                warp::copy(D_reg_bf_tmp, D_reg_fl);
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[epi%C::NUM_D_TILES], D_reg_bf_tmp);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D_scratch, output_tiles.D[epi%C::NUM_D_TILES], {0, 0});
            }

            // ============ PHASE 1b: Batch-load subtiles 4..7 → FP32 regs ============
            // Keep FP32 to avoid bf16→FP32 round-trip in Phase 2.
            subtile_rt D_reg_fl_stash[HALF];
            #pragma unroll
            for (int epi = 0; epi < HALF; epi++) {
                warpgroup::load_async(D_reg_fl_stash[epi], out_tm.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * (epi + HALF)));
                warp::mul(D_reg_fl_stash[epi], D_reg_fl_stash[epi], global_scale);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            // ============ PHASE 2: Deferred CCE for subtiles 4..7 (overlaps producer MMA) ============
            #pragma unroll
            for (int epi = 0; epi < HALF; epi++) {
                subtile_rt &D_reg_fl = D_reg_fl_stash[epi];

                // Target extraction — skip if no target can land in this subtile
                {
                    int col_start = col_block_idx * C::Nb + (epi + HALF) * SUBTILE_COLS;
                    int col_end = col_start + SUBTILE_COLS;
                    if (col_start < g.N && col_end > 0) {
                        int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                        int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
                        warp::apply(D_reg_fl, D_reg_fl, [&](int local_row, int local_col, float val) -> float {
                            int global_row = warp_row_base + local_row;
                            int global_col = col_start + local_col;
                            if (global_row < g.M && global_col < g.N) {
                                int64_t target = g.targets[global_row];
                                if ((int)target == global_col) {
                                    atomicAdd(&g.neg_correct_logit[global_row], -val);
                                }
                            }
                            return val;
                        });
                    }
                }

                // Online logsumexp — no -INFINITY guard (running state is finite from Phase 1a)
                col_vec_t tile_max;
                warp::row_max(tile_max, D_reg_fl);
                warp::sub_row(D_reg_fl, D_reg_fl, tile_max);
                warp::exp(D_reg_fl, D_reg_fl);
                col_vec_t tile_sumexp;
                warp::row_sum(tile_sumexp, D_reg_fl);
                #pragma unroll
                for (int i = 0; i < col_vec_t::outer_dim; i++) {
                    #pragma unroll
                    for (int j = 0; j < col_vec_t::inner_dim; j++) {
                        float old_max_x = running_max[i][j].x;
                        float new_max_x = fmaxf(old_max_x, tile_max[i][j].x);
                        float old_max_y = running_max[i][j].y;
                        float new_max_y = fmaxf(old_max_y, tile_max[i][j].y);
                        running_sumexp[i][j].x = running_sumexp[i][j].x * __expf(old_max_x - new_max_x) +
                                                 tile_sumexp[i][j].x * __expf(tile_max[i][j].x - new_max_x);
                        running_sumexp[i][j].y = running_sumexp[i][j].y * __expf(old_max_y - new_max_y) +
                                                 tile_sumexp[i][j].y * __expf(tile_max[i][j].y - new_max_y);
                        running_max[i][j].x = new_max_x;
                        running_max[i][j].y = new_max_y;
                    }
                }

                // TMA store for pacing — convert to bf16 for store
                bf16_subtile_rt D_reg_bf_tmp;
                warp::copy(D_reg_bf_tmp, D_reg_fl);
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[epi%C::NUM_D_TILES], D_reg_bf_tmp);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(g.D_scratch, output_tiles.D[epi%C::NUM_D_TILES], {0, 0});
            }

            // ============ PHASE 3: Finalize LSE ============
            const int lane_id = threadIdx.x % 32;
            const int warpid_in_wg = warpgroup::warpid();
            int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
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

            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace nvfp4_cce
