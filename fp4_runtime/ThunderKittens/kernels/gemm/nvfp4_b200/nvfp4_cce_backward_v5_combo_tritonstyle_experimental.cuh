#pragma once
// ================================================================
// NVFP4 CCE Backward v5 Triton-style one-pass combo experiment
//
// Single-CTA schedule:
//   1. Recompute FP4 logits for one (128 rows, 128 vocab cols) block
//      as two legal one-CTA 64-col halves in TMEM.
//   2. Form the softmax-gradient BF16 tile once per 64-col half.
//   3. Immediately consume that BF16 grad tile for:
//        dE += grad @ C_bf16
//        dC^T += E_bf16^T @ grad
//
// This keeps the Triton-style "compute grad once, update both outputs"
// schedule while staying on legal single-CTA TCGen shapes.
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v5_combo_tritonstyle_experimental {

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true, int _EPI_PIPE_DEPTH = 4>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_EPI_PIPE_DEPTH > 0);

    static constexpr int CLUSTER_SIZE = 1;
    static constexpr bool USE_PDL = false;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr int ROW_TILE = 128;
    static constexpr int Nb = 128;          // logical vocab block per CTA
    static constexpr int LOCAL_N = Nb / 2;  // one-CTA legal half-block
    static constexpr int Kb = 256;          // logits reduction
    static constexpr int Nb_out = 128;      // K-tile for output updates
    static_assert(LOCAL_N % EPI_PIPE_DEPTH == 0);
    static_assert(Nb_out % EPI_PIPE_DEPTH == 0);

    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int STORE_PIPE_STAGES = (EPI_PIPE_DEPTH < 2 ? EPI_PIPE_DEPTH : 2);
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::ROW_TILE, C::Kb / 2>;
    using A_sc_tile = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::LOCAL_N, C::Kb / 2>;
    using B_sc_tile = st_hf<4, 256, false>;

    using Grad_row_tile = st_bf<C::ROW_TILE, C::LOCAL_N>;
    using Grad_subtile = st_bf<C::ROW_TILE, C::LOCAL_N / C::EPI_PIPE_DEPTH>;
    using C_tile = st_bf<C::LOCAL_N, C::Nb_out>;
    using E_tile = st_bf<C::ROW_TILE, C::Nb_out>;
    using dE_tile = st_bf<C::ROW_TILE, C::Nb_out / C::EPI_PIPE_DEPTH>;
    using dC_tile = st_bf<C::LOCAL_N / C::EPI_PIPE_DEPTH, C::Nb_out>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl = gl<half, 1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float, 1, 1, 1, 1>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl = gl<half, 1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float, 1, 1, 1, 1>;

    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;
    using E_gl = gl<bf16, 1, 1, -1, -1, E_tile>;
    using dE_gl = gl<bf16, 1, 1, -1, -1, dE_tile>;
    using dC_gl = gl<bf16, 1, 1, -1, -1, dC_tile>;

    A_fp4x2_gl A;
    A_sc_gl A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl B;
    B_sc_gl B_sc;
    B_sc_global_gl B_sc_global;

    E_gl E_bf16;
    C_gl C_bf16;
    dE_gl dE_out;
    dC_gl dC_out;

    const float *lse;
    const int64_t *targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;
    int K;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B[2];
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[2][C::B_SC_SIZE];
    };
    struct grad_tiles_t {
        Grad_row_tile G_row;
    };
    struct phase3_tiles_t {
        C_tile C_half;
        E_tile E_rows;
    };
    struct output_tiles_t {
        Grad_subtile logits;
        dE_tile dE[C::STORE_PIPE_STAGES];
        dC_tile dC[C::STORE_PIPE_STAGES];
    };

    __host__ inline dim3 grid() const {
        const int num_row_blocks = A.rows() / C::ROW_TILE;
        const int num_col_blocks = N / C::Nb;
        return dim3(num_col_blocks, num_row_blocks);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int phase1_smem =
            sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
            sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024;
        constexpr int grad_smem = sizeof(grad_tiles_t);
        constexpr int p3_smem = sizeof(phase3_tiles_t);
        constexpr int out_smem = sizeof(output_tiles_t);
        constexpr int total = phase1_smem + grad_smem + p3_smem + out_smem;
        static_assert(total <= MAX_SHARED_MEMORY - 1024);
        return total;
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
        g.E_bf16.template prefetch_tma<typename G::E_tile>();
        g.C_bf16.template prefetch_tma<typename G::C_tile>();
        g.dE_out.template prefetch_tma<typename G::dE_tile>();
        g.dC_out.template prefetch_tma<typename G::dC_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int lane_id = threadIdx.x % WARP_THREADS;
    const int row_block_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;
    const int num_iters_per_block = 2 * g.A.cols() / C::Kb;
    const int num_k_blocks = g.K / C::Nb_out;
    uint32_t load_phasebits = 0xFFFF0000;
    int load_stage = 0;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::grad_tiles_t &grad_tiles = sm_allocator.allocate<typename G::grad_tiles_t>();
    typename G::phase3_tiles_t &phase3_tiles = sm_allocator.allocate<typename G::phase3_tiles_t>();
    typename G::output_tiles_t &output_tiles = sm_allocator.allocate<typename G::output_tiles_t>();

    tensor_allocator<1, 1, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore phase1_outputs_arrived;
    __shared__ semaphore c_tiles_arrived;
    __shared__ semaphore e_tiles_arrived;
    __shared__ semaphore de_outputs_arrived;
    __shared__ semaphore dc_outputs_arrived;
    __shared__ float filter_max_smem[WARPGROUP_WARPS];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(phase1_outputs_arrived, 0, 1);
        init_semaphore(c_tiles_arrived, 0, 1);
        init_semaphore(e_tiles_arrived, 0, 1);
        init_semaphore(de_outputs_arrived, 0, 1);
        init_semaphore(dc_outputs_arrived, 0, 1);
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS * C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            everyone::tma::cluster::wait();
            for (int i = 0; i < num_iters_per_block; ++i) {
                wait(inputs_finished[load_stage], get_phasebit<1>(load_phasebits, load_stage));
                tma::load_async(input_tiles[load_stage].A, g.A, {row_block_idx, i}, tiles_arrived[load_stage]);
                tma::load_async(input_tiles[load_stage].B[0], g.B, {col_block_idx * 2 + 0, i}, tiles_arrived[load_stage]);
                tma::load_async(input_tiles[load_stage].B[1], g.B, {col_block_idx * 2 + 1, i}, tiles_arrived[load_stage]);
                update_phasebit<1>(load_phasebits, load_stage);
                load_stage = (load_stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        } else if (warp_id == 2) {
            int scale_stage = 0;
            everyone::tma::cluster::wait();
            for (int i = 0; i < num_iters_per_block; ++i) {
                wait(inputs_finished[scale_stage], get_phasebit<1>(load_phasebits, scale_stage));
                tma::load_async(input_scales[scale_stage].A, g.A_sc, {row_block_idx, i, 0}, scales_arrived[scale_stage]);
                #pragma unroll
                for (int sc = 0; sc < C::B_SC_SIZE; ++sc) {
                    tma::load_async(input_scales[scale_stage].B[0][sc], g.B_sc, {col_block_idx * 2 + 0, i, sc}, scales_arrived[scale_stage]);
                    tma::load_async(input_scales[scale_stage].B[1][sc], g.B_sc, {col_block_idx * 2 + 1, i, sc}, scales_arrived[scale_stage]);
                }
                update_phasebit<1>(load_phasebits, scale_stage);
                scale_stage = (scale_stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        } else if (warp_id == 0) {
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            auto phase1_tm_0 = tm_allocator.template allocate<full_tt_fl<C::LOCAL_N>>(0);
            auto phase1_tm_1 = tm_allocator.template allocate<full_tt_fl<C::LOCAL_N>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE>>(
                256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE>>(
                256 + 4 * C::MMA_PER_TILE * (C::LOAD_PIPE_DEPTH + 1));

            constexpr uint32_t SCALE_BYTES =
                sizeof(typename G::A_sc_tile) + 2 * C::B_SC_SIZE * sizeof(typename G::B_sc_tile);
            constexpr uint32_t TILE_BYTES =
                sizeof(typename G::A_fp4x2_tile) + 2 * sizeof(typename G::B_fp4x2_tile);

            int stage = 0;
            tensor_after_thread_sync();
            for (int i = 0; i < num_iters_per_block; ++i) {
                tma::expect_bytes(scales_arrived[stage], SCALE_BYTES);
                wait(scales_arrived[stage], get_phasebit<0>(load_phasebits, stage));

                #pragma unroll
                for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                    auto A_sc_tm_sub = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                        stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &A_sc_sm_sub =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(A_sc_tm_sub, A_sc_sm_sub);

                    auto B_sc_tm_sub_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    auto &B_sc_sm_sub_0 =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].B[0][0].data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(B_sc_tm_sub_0, B_sc_sm_sub_0);

                    auto B_sc_tm_sub_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    auto &B_sc_sm_sub_1 =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].B[1][0].data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(B_sc_tm_sub_1, B_sc_sm_sub_1);
                }

                tma::expect_bytes(tiles_arrived[stage], TILE_BYTES);
                wait(tiles_arrived[stage], get_phasebit<0>(load_phasebits, stage));
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(
                    stage * C::MMA_PER_TILE * 16);
                auto B_sc_tm_tile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(0);
                auto B_sc_tm_tile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(0);

                if (i == 0) {
                    mm_ABt(phase1_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                    mm_ABt(phase1_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                } else {
                    mma_ABt(phase1_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                    mma_ABt(phase1_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                }
                kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);
                update_phasebit<0>(load_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
            tensor_commit<1>(phase1_outputs_arrived);
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        const bool issue_leader = (warpgroup::warpid() == 0 && lane_id == 0);

        auto phase1_tm_0 = tm_allocator.template allocate<full_tt_fl<C::LOCAL_N>>(0);
        auto phase1_tm_1 = tm_allocator.template allocate<full_tt_fl<C::LOCAL_N>>(128);
        auto de_tm = tm_allocator.template allocate<full_tt_fl<C::Nb_out>>(256);
        auto dc_tm = tm_allocator.template allocate<full_tt_fl<C::LOCAL_N>>(384);
        wait(phase1_outputs_arrived, 0);
        tensor_after_thread_sync();

        constexpr int SUBTILE_COLS = C::LOCAL_N / C::EPI_PIPE_DEPTH;
        constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
        using logits_rt = rt_fl<C::ROW_TILE / WARPGROUP_WARPS, SUBTILE_COLS>;
        using logits_rt_bf = rt_bf<C::ROW_TILE / WARPGROUP_WARPS, SUBTILE_COLS>;
        using de_rt = rt_fl<C::ROW_TILE / WARPGROUP_WARPS, C::Nb_out / C::EPI_PIPE_DEPTH>;
        using de_rt_bf = rt_bf<C::ROW_TILE / WARPGROUP_WARPS, C::Nb_out / C::EPI_PIPE_DEPTH>;

        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}];
        const int tile_row_base = row_block_idx * C::ROW_TILE;
        const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::ROW_TILE / WARPGROUP_WARPS);
        const uint32_t grad_row_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(&grad_tiles.G_row.data[0]));
        const uint32_t logits_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.logits.data[0]));
        const int wg_thread = warpgroup::warpid() * WARP_THREADS + lane_id;

        int my_targets_x[logits_rt::height];
        int my_targets_y[logits_rt::height];
        float my_lse_x[logits_rt::height];
        float my_lse_y[logits_rt::height];
        #pragma unroll
        for (int i = 0; i < logits_rt::height; ++i) {
            const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
            const int global_row_y = global_row_x + 8;
            my_targets_x[i] = (global_row_x < g.M) ? static_cast<int>(g.targets[global_row_x]) : -1;
            my_targets_y[i] = (global_row_y < g.M) ? static_cast<int>(g.targets[global_row_y]) : -1;
            my_lse_x[i] = (global_row_x < g.M) ? g.lse[global_row_x] : INFINITY;
            my_lse_y[i] = (global_row_y < g.M) ? g.lse[global_row_y] : INFINITY;
        }

        int c_phase = 0;
        int e_phase = 0;
        int de_phase = 0;
        int dc_phase = 0;

        for (int half = 0; half < 2; ++half) {
            auto &phase1_tm = (half == 0) ? phase1_tm_0 : phase1_tm_1;
            logits_rt D_pipe[2];
            logits_rt_bf D_bf;
            float filter_local_max = 0.0f;

            warpgroup::load_async(
                D_pipe[0],
                phase1_tm.template subtile<full_tt_fl<C::LOCAL_N / C::EPI_PIPE_DEPTH>>(0, 0));
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            int cur_slot = 0;

            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                const int next_slot = cur_slot ^ 1;
                if constexpr (C::EPI_PIPE_DEPTH > 1) {
                    if (epi + 1 < C::EPI_PIPE_DEPTH) {
                        warpgroup::load_async(
                            D_pipe[next_slot],
                            phase1_tm.template subtile<full_tt_fl<C::LOCAL_N / C::EPI_PIPE_DEPTH>>(
                                0, (epi + 1) * (C::LOCAL_N / C::EPI_PIPE_DEPTH)));
                    }
                }

                auto &D_fl = D_pipe[cur_slot];
                warp::mul(D_fl, D_fl, global_scale);

                const int col_start =
                    col_block_idx * C::Nb + half * C::LOCAL_N + epi * SUBTILE_COLS;

                #pragma unroll
                for (int i = 0; i < logits_rt::height; ++i) {
                    const float lse_x = my_lse_x[i];
                    const float lse_y = my_lse_y[i];
                    #pragma unroll
                    for (int j = 0; j < logits_rt::width; ++j) {
                        #pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            const float lse_val = (kk % 2 == 0) ? lse_x : lse_y;
                            D_fl.tiles[i][j].data[kk].x = __expf(D_fl.tiles[i][j].data[kk].x - lse_val);
                            D_fl.tiles[i][j].data[kk].y = __expf(D_fl.tiles[i][j].data[kk].y - lse_val);
                        }
                    }
                }

                #pragma unroll
                for (int i = 0; i < logits_rt::height; ++i) {
                    const int tgt_x = my_targets_x[i];
                    if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
                        const int local_col = tgt_x - col_start;
                        const int j_idx = local_col / 16;
                        const int within_tile = local_col % 16;
                        const int k_half = within_tile / 8;
                        const int pair_pos = (within_tile % 8) / 2;
                        if ((lane_id % 4) == pair_pos) {
                            const int k_idx = k_half * 2;
                            if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                            else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                        }
                    }
                    const int tgt_y = my_targets_y[i];
                    if (tgt_y >= col_start && tgt_y < col_start + SUBTILE_COLS) {
                        const int local_col = tgt_y - col_start;
                        const int j_idx = local_col / 16;
                        const int within_tile = local_col % 16;
                        const int k_half = within_tile / 8;
                        const int pair_pos = (within_tile % 8) / 2;
                        if ((lane_id % 4) == pair_pos) {
                            const int k_idx = k_half * 2 + 1;
                            if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                            else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                        }
                    }
                }

                #pragma unroll
                for (int i = 0; i < logits_rt::height; ++i) {
                    const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                    const int global_row_y = global_row_x + 8;
                    #pragma unroll
                    for (int j = 0; j < logits_rt::width; ++j) {
                        #pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            if (kk % 2 == 0 && global_row_x >= g.M) {
                                D_fl.tiles[i][j].data[kk].x = 0.0f;
                                D_fl.tiles[i][j].data[kk].y = 0.0f;
                            }
                            if (kk % 2 == 1 && global_row_y >= g.M) {
                                D_fl.tiles[i][j].data[kk].x = 0.0f;
                                D_fl.tiles[i][j].data[kk].y = 0.0f;
                            }
                        }
                    }
                }

                if (g.filter_eps > 0.0f) {
                    #pragma unroll
                    for (int i = 0; i < logits_rt::height; ++i) {
                        #pragma unroll
                        for (int j = 0; j < logits_rt::width; ++j) {
                            #pragma unroll
                            for (int kk = 0; kk < 4; ++kk) {
                                filter_local_max = fmaxf(filter_local_max, fabsf(D_fl.tiles[i][j].data[kk].x));
                                filter_local_max = fmaxf(filter_local_max, fabsf(D_fl.tiles[i][j].data[kk].y));
                            }
                        }
                    }
                }
                warp::mul(D_fl, D_fl, g.grad_scale);
                warp::copy(D_bf, D_fl);
                warpgroup::sync(1);
                warpgroup::store(output_tiles.logits, D_bf);
                warpgroup::sync(1);

                if (wg_thread < C::ROW_TILE) {
                    #pragma unroll
                    for (int col = 0; col < SUBTILE_COLS; ++col) {
                        bf16 value_bf;
                        move<bf16>::lds(value_bf, G::Grad_subtile::idx(logits_base, {wg_thread, col}));
                        move<bf16>::sts(
                            G::Grad_row_tile::idx(grad_row_base, {wg_thread, epi * SUBTILE_COLS + col}),
                            value_bf);
                    }
                }

                warpgroup::sync(1);
                if constexpr (C::EPI_PIPE_DEPTH > 1) {
                    if (epi + 1 < C::EPI_PIPE_DEPTH) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        cur_slot = next_slot;
                    }
                }
            }

            bool half_is_filtered = false;
            if (g.filter_eps > 0.0f) {
                #pragma unroll
                for (int offset = WARP_THREADS / 2; offset > 0; offset >>= 1) {
                    filter_local_max = fmaxf(filter_local_max, __shfl_xor_sync(0xFFFFFFFF, filter_local_max, offset));
                }
                if (lane_id == 0) filter_max_smem[warpgroup::warpid()] = filter_local_max;
                warpgroup::sync(1);

                float global_max = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPGROUP_WARPS; ++w) {
                    global_max = fmaxf(global_max, filter_max_smem[w]);
                }
                half_is_filtered = (global_max < g.filter_eps);
                warpgroup::sync(1);
            }

            if (half_is_filtered) {
                continue;
            }

            warpgroup::sync(1);
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

            for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                if (issue_leader) {
                    tma::load_async(phase3_tiles.C_half, g.C_bf16, {col_block_idx * 2 + half, k_block_idx}, c_tiles_arrived);
                }
                if (issue_leader) {
                    tma::expect_bytes(c_tiles_arrived, sizeof(typename G::C_tile));
                }
                wait(c_tiles_arrived, c_phase);
                c_phase ^= 1;
                warpgroup::sync(1);

                if (issue_leader) {
                    mm_AB(de_tm, grad_tiles.G_row, phase3_tiles.C_half);
                    tensor_commit<1>(de_outputs_arrived);
                }
                wait(de_outputs_arrived, de_phase);
                de_phase ^= 1;

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    const int smem_slot = epi % C::STORE_PIPE_STAGES;
                    de_rt D_reg_fl;
                    de_rt_bf D_reg_bf;
                    warpgroup::load_async(
                        D_reg_fl,
                        de_tm.template subtile<full_tt_fl<C::Nb_out / C::EPI_PIPE_DEPTH>>(
                            0, epi * (C::Nb_out / C::EPI_PIPE_DEPTH)));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    if constexpr (C::EPI_PIPE_DEPTH > C::STORE_PIPE_STAGES) {
                        if (epi >= C::STORE_PIPE_STAGES) {
                            warpgroup::tma::store_async_read_wait<C::STORE_PIPE_STAGES - 1>();
                        }
                    }
                    warpgroup::sync(1);
                    warp::copy(D_reg_bf, D_reg_fl);
                    warpgroup::store(output_tiles.dE[smem_slot], D_reg_bf);
                    warpgroup::sync(1);
                    warpgroup::tma::store_add_async(
                        g.dE_out, output_tiles.dE[smem_slot],
                        {row_block_idx, k_block_idx * C::EPI_PIPE_DEPTH + epi});
                }
                warpgroup::sync(1);
                tensor_after_thread_sync();
                if constexpr (C::EPI_PIPE_DEPTH > C::STORE_PIPE_STAGES) {
                    warpgroup::tma::store_async_read_wait<0>();
                }

                if (issue_leader) {
                    tma::load_async(phase3_tiles.E_rows, g.E_bf16, {row_block_idx, k_block_idx}, e_tiles_arrived);
                }
                if (issue_leader) {
                    tma::expect_bytes(e_tiles_arrived, sizeof(typename G::E_tile));
                }
                wait(e_tiles_arrived, e_phase);
                e_phase ^= 1;
                warpgroup::sync(1);

                if (issue_leader) {
                    mm_AtB(dc_tm, phase3_tiles.E_rows, grad_tiles.G_row);
                    tensor_commit<1>(dc_outputs_arrived);
                }
                wait(dc_outputs_arrived, dc_phase);
                dc_phase ^= 1;

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    const int smem_slot = epi % C::STORE_PIPE_STAGES;
                    const uint32_t dC_base =
                        static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.dC[smem_slot].data[0]));
                    logits_rt D_reg_fl;
                    logits_rt_bf D_reg_bf;
                    warpgroup::load_async(
                        D_reg_fl,
                        dc_tm.template subtile<full_tt_fl<C::LOCAL_N / C::EPI_PIPE_DEPTH>>(
                            0, epi * (C::LOCAL_N / C::EPI_PIPE_DEPTH)));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    if constexpr (C::EPI_PIPE_DEPTH > C::STORE_PIPE_STAGES) {
                        if (epi >= C::STORE_PIPE_STAGES) {
                            warpgroup::tma::store_async_read_wait<C::STORE_PIPE_STAGES - 1>();
                        }
                    }
                    warpgroup::sync(1);
                    warp::copy(D_reg_bf, D_reg_fl);
                    warpgroup::store(output_tiles.logits, D_reg_bf);
                    warpgroup::sync(1);

                    if (wg_thread < C::ROW_TILE) {
                        #pragma unroll
                        for (int col = 0; col < C::LOCAL_N / C::EPI_PIPE_DEPTH; ++col) {
                            bf16 value_bf;
                            move<bf16>::lds(value_bf, G::Grad_subtile::idx(logits_base, {wg_thread, col}));
                            move<bf16>::sts(G::dC_tile::idx(dC_base, {col, wg_thread}), value_bf);
                        }
                    }
                    warpgroup::sync(1);

                    warpgroup::tma::store_add_async(
                        g.dC_out, output_tiles.dC[smem_slot],
                        {col_block_idx * (C::Nb / (C::LOCAL_N / C::EPI_PIPE_DEPTH)) +
                             half * (C::LOCAL_N / (C::LOCAL_N / C::EPI_PIPE_DEPTH)) + epi,
                         k_block_idx});
                }
                warpgroup::tma::store_async_read_wait<0>();
                warpgroup::sync(1);
                tensor_after_thread_sync();
            }
        }

        tma::store_async_wait();
        warpgroup::sync(1);
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
}

} // namespace nvfp4_cce_backward_v5_combo_tritonstyle_experimental
