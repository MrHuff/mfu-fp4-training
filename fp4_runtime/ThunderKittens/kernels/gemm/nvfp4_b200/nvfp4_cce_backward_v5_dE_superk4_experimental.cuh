#pragma once
// ================================================================
// NVFP4 CCE Backward v5 dE experimental sandbox
//
// Grid ownership: (row_block, k_superblock_of_2)
// Outer reduction: sweep vocab blocks
// Phase 1: recompute logits tile for (row_block, vocab_block)
// Phase 2: form logits-gradient G and row-quantize it into SMEM
// Phase 3: reuse one staged G_row across 2 adjacent k blocks
//
// Experimental warpgroup split:
//  - producer WG: TMA only
//  - math WG: phase 1 / phase 3 tensor-core issue + dE accumulation
//  - epilogue WG: softmax + FP4 staging
//
// `G` never touches HBM. Partial dE tiles accumulate on-chip in the math
// warpgroup registers and store once at the end of the vocab sweep.
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v5_dE_superk4_experimental {

__device__ __forceinline__ uint8_t quantize_fp4_pair_v5_dE(float v0, float v1, float rcp_scale) {
    const float2 scaled = {v0 * rcp_scale, v1 * rcp_scale};
    return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
}

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int EPILOGUE_WARPGROUPS = 1;
    static constexpr int MATH_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int CONSUMER_WARPGROUPS = EPILOGUE_WARPGROUPS + MATH_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int EPILOGUE_WARPGROUP_ID = 0;
    static constexpr int MATH_WARPGROUP_ID = EPILOGUE_WARPGROUP_ID + EPILOGUE_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUP_ID = MATH_WARPGROUP_ID + MATH_WARPGROUPS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = 2;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;
    static constexpr int K_SUPERBLOCK = 2;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int Nb_out = 128;

    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int P3_MMA_PER_TILE = Nb / 64;
    static constexpr int NUM_D_TILES = 2;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    using P3_B_fp4x2_tile = st_fp4e2m1_2<C::Nb_out/2, C::Nb/2>;
    using P3_B_sc_tile    = st_hf<2, 256, false>;

    using G_fp4_row_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;
    using G_sc_row_tile  = st_hf<2, 256, false>;

    using Out_tile = st_bf<C::Mb/2, C::Nb_out / C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;

    using P3_B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, P3_B_fp4x2_tile>;
    using P3_B_sc_gl        = gl<half,       1, -1, -1, 256, P3_B_sc_tile>;
    using P3_B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using Out_gl            = gl<bf16,       1,  1, -1, -1, Out_tile>;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;

    P3_B_fp4x2_gl     C_col;
    P3_B_sc_gl        C_col_sc;
    P3_B_sc_global_gl C_col_sc_global;
    Out_gl            dE_out;

    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;
    int K;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct output_tiles_t {
        Out_tile D[C::NUM_D_TILES];
    };
    struct fp4_staging_t {
        G_fp4_row_tile G_row;
        G_sc_row_tile  G_row_sc;
    };
    struct p3_tiles_t {
        P3_B_fp4x2_tile B;
    };
    struct p3_scales_t {
        P3_B_sc_tile B_sc;
    };

    __host__ inline dim3 grid() const {
        const int num_row_blocks = A.rows() / C::Mb;
        const int num_k_blocks = dE_out.cols() / C::Nb_out;
        const int num_k_superblocks = (num_k_blocks + C::K_SUPERBLOCK - 1) / C::K_SUPERBLOCK;
        int total = num_row_blocks * num_k_superblocks;
        int grid_size = min(total, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int phase1_smem = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(output_tiles_t);
        constexpr int fp4_smem = sizeof(fp4_staging_t) * 2;
        constexpr int p3_smem = (sizeof(p3_tiles_t) + 1024 + sizeof(p3_scales_t) + 1024) * C::K_SUPERBLOCK;
        constexpr int total = phase1_smem + fp4_smem + p3_smem;
        static_assert(total <= MAX_SHARED_MEMORY - 1024);
        return total;
    }
};

template <typename C>
__device__ inline void kernel(const globals<C>& g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.C_col.template prefetch_tma<typename G::P3_B_fp4x2_tile>();
        g.C_col_sc.template prefetch_tma<typename G::P3_B_sc_tile>();
        g.dE_out.template prefetch_tma<typename G::Out_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;

    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_vocab_blocks = g.B.rows() / C::Nb;
    const int num_k_blocks = g.dE_out.cols() / C::Nb_out;
    const int num_k_superblocks = (num_k_blocks + C::K_SUPERBLOCK - 1) / C::K_SUPERBLOCK;
    const int num_blocks = num_row_blocks * num_k_superblocks;
    const int num_iters_per_vocab = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_k_superblocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::output_tiles_t &output_tiles = sm_allocator.allocate<G::output_tiles_t>();
    typename G::fp4_staging_t (&fp4_staging)[2] = sm_allocator.allocate<G::fp4_staging_t, 2>();
    typename G::p3_tiles_t    (&p3_tiles)[C::K_SUPERBLOCK] = sm_allocator.allocate<G::p3_tiles_t, C::K_SUPERBLOCK>();
    typename G::p3_scales_t   (&p3_scales)[C::K_SUPERBLOCK] = sm_allocator.allocate<G::p3_scales_t, C::K_SUPERBLOCK>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore p3_tiles_arrived[C::K_SUPERBLOCK];
    __shared__ semaphore p3_scales_arrived[C::K_SUPERBLOCK];
    __shared__ semaphore p3_inputs_finished[C::K_SUPERBLOCK];
    __shared__ semaphore p3_outputs_arrived[C::K_SUPERBLOCK];
    __shared__ semaphore p3_outputs_finished;
    __shared__ semaphore fp4_ready;
    __shared__ semaphore fp4_ready_cluster;
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
        #pragma unroll
        for (int ksub = 0; ksub < C::K_SUPERBLOCK; ++ksub) {
            init_semaphore(p3_tiles_arrived[ksub], 0, 1);
            init_semaphore(p3_scales_arrived[ksub], 0, 1);
            init_semaphore(p3_inputs_finished[ksub], 0, 1);
            init_semaphore(p3_outputs_arrived[ksub], 0, 1);
        }
        init_semaphore(p3_outputs_finished, 0, C::CLUSTER_SIZE);
        init_semaphore(fp4_ready, 1, 0);
        init_semaphore(fp4_ready_cluster, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == C::PRODUCER_WARPGROUP_ID && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                const int k_superblock_idx = idx_within_supergroup / rows_in_supergroup;
                const int k_block_base = C::K_SUPERBLOCK * k_superblock_idx;

                for (int vocab_block_idx = 0; vocab_block_idx < num_vocab_blocks; ++vocab_block_idx) {
                    for (int i = 0; i < num_iters_per_vocab; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        tma::cluster::load_async(
                            input_tiles[stage].A, g.A,
                            {row_block_idx * 2 + cta_id, i},
                            tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        tma::cluster::load_async(
                            input_tiles[stage].B, g.B,
                            {vocab_block_idx * 2 + cta_id, i},
                            tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }

                    wait(fp4_ready, get_phasebit<1>(phasebits, 5));
                    #pragma unroll
                    for (int ksub = 0; ksub < C::K_SUPERBLOCK; ++ksub) {
                        const int k_block_idx = k_block_base + ksub;
                        if (k_block_idx >= num_k_blocks) continue;
                        wait(p3_inputs_finished[ksub], get_phasebit<1>(phasebits, 6 + ksub));
                        tma::cluster::load_async(
                            p3_tiles[ksub].B, g.C_col,
                            {k_block_idx * 2 + cta_id, vocab_block_idx},
                            p3_tiles_arrived[ksub], (uint16_t)(1 << cta_id), 0);
                        update_phasebit<1>(phasebits, 6 + ksub);
                    }
                    update_phasebit<1>(phasebits, 5);
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                const int k_superblock_idx = idx_within_supergroup / rows_in_supergroup;
                const int k_block_base = C::K_SUPERBLOCK * k_superblock_idx;

                for (int vocab_block_idx = 0; vocab_block_idx < num_vocab_blocks; ++vocab_block_idx) {
                    for (int i = 0; i < num_iters_per_vocab; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        tma::cluster::load_async(
                            input_scales[stage].A, g.A_sc,
                            {row_block_idx * 2 + cta_id, i, 0},
                            scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        if (cta_id == 0) {
                            tma::cluster::load_async(
                                input_scales[stage].B[0], g.B_sc,
                                {vocab_block_idx, i, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }

                    wait(fp4_ready, get_phasebit<1>(phasebits, 5));
                    #pragma unroll
                    for (int ksub = 0; ksub < C::K_SUPERBLOCK; ++ksub) {
                        const int k_block_idx = k_block_base + ksub;
                        if (k_block_idx >= num_k_blocks) continue;
                        wait(p3_inputs_finished[ksub], get_phasebit<1>(phasebits, 6 + ksub));
                        tma::cluster::load_async(
                            p3_scales[ksub].B_sc, g.C_col_sc,
                            {k_block_idx, vocab_block_idx, 0},
                            p3_scales_arrived[ksub], (uint16_t)(1 << cta_id), 0);
                        update_phasebit<1>(phasebits, 6 + ksub);
                    }
                    update_phasebit<1>(phasebits, 5);
                }
            }
        }
    } else if (warpgroup_id == C::MATH_WARPGROUP_ID) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
        auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
        auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
            256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
        constexpr int p3_sc_offset = 256;
        auto p3_A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::P3_MMA_PER_TILE>>(p3_sc_offset);
        auto p3_B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::P3_MMA_PER_TILE>>(
            p3_sc_offset + 4 * C::P3_MMA_PER_TILE);

        using out_rt = rt_fl<C::Mb / 8, C::Nb_out / C::EPI_PIPE_DEPTH>;
        constexpr float kFp4Max = 6.0f;
        constexpr float kE4M3Max = 448.0f;
        const float fp4_row_sg = fmaxf(g.grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
        const float p3_scale = fp4_row_sg * g.C_col_sc_global[{0}];

        uint32_t issue_stage = 0;
        uint32_t issue_phasebits = 0xFFFF0000;
        uint32_t acc_phasebits = 0xFFFF0000;
        int phase = 0;

        auto do_phase1 = [&](auto &accum) {
            for (int i = 0; i < num_iters_per_vocab; ++i) {
                tma::expect_bytes(scales_arrived[issue_stage], 2 * sizeof(typename G::input_scales_t));
                wait(scales_arrived[issue_stage], get_phasebit<0>(issue_phasebits, issue_stage));
                #pragma unroll
                for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                    auto A_sc_tm_sub = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                        issue_stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &A_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                        reinterpret_cast<uint64_t>(&input_scales[issue_stage].A.data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async2(A_sc_tm_sub, A_sc_sm_sub);

                    auto B_sc_tm_sub = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(
                        issue_stage * C::MMA_PER_TILE * 32 + ii * 16);
                    auto &B_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                        reinterpret_cast<uint64_t>(&input_scales[issue_stage].B[0].data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async2(B_sc_tm_sub, B_sc_sm_sub);
                }

                tma::expect_bytes(tiles_arrived[issue_stage], 2 * sizeof(typename G::input_tiles_t));
                wait(tiles_arrived[issue_stage], get_phasebit<0>(issue_phasebits, issue_stage));
                if (i == 0) {
                    mm2_ABt(
                        accum, input_tiles[issue_stage].A, input_tiles[issue_stage].B,
                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(issue_stage * C::MMA_PER_TILE * 16),
                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(issue_stage * C::MMA_PER_TILE * 32),
                        inputs_finished[issue_stage]);
                } else {
                    mma2_ABt(
                        accum, input_tiles[issue_stage].A, input_tiles[issue_stage].B,
                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(issue_stage * C::MMA_PER_TILE * 16),
                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(issue_stage * C::MMA_PER_TILE * 32),
                        inputs_finished[issue_stage]);
                }
                update_phasebit<0>(issue_phasebits, issue_stage);
                issue_stage = (issue_stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        };

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            const int supergroup_idx = block_idx / num_blocks_per_supergroup;
            const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            const int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            const int k_superblock_idx = idx_within_supergroup / rows_in_supergroup;
            const int k_block_base = C::K_SUPERBLOCK * k_superblock_idx;

            out_rt D_acc[C::K_SUPERBLOCK][C::EPI_PIPE_DEPTH];

            for (int vocab_block_idx = 0; vocab_block_idx < num_vocab_blocks; ++vocab_block_idx) {
                auto &p3_accum = (phase == 0) ? out_tm_0 : out_tm_1;
                int last_valid_ksub = -1;
                #pragma unroll
                for (int ksub = 0; ksub < C::K_SUPERBLOCK; ++ksub) {
                    if (k_block_base + ksub < num_k_blocks) last_valid_ksub = ksub;
                }

                if (warp::elect_leader() && cta_id == 0 && warpgroup::warpid() == 0) {
                    wait(outputs_finished, get_phasebit<1>(issue_phasebits, 0));
                    tensor_after_thread_sync();

                    do_phase1(p3_accum);
                    tensor_commit<2>(outputs_arrived);

                    wait(fp4_ready_cluster, get_phasebit<0>(issue_phasebits, 5));
                    tensor_after_thread_sync();

                    #pragma unroll
                    for (int ii = 0; ii < C::P3_MMA_PER_TILE; ++ii) {
                        auto p3_A_sc_sub = p3_A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                        auto &G_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&fp4_staging[phase].G_row_sc.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(p3_A_sc_sub, G_sc_sm_sub);
                    }

                }

                #pragma unroll
                for (int ksub = 0; ksub < C::K_SUPERBLOCK; ++ksub) {
                    const int k_block_idx = k_block_base + ksub;
                    if (k_block_idx >= num_k_blocks) continue;

                    if (warp::elect_leader() && cta_id == 0 && warpgroup::warpid() == 0) {
                        tma::expect_bytes(p3_scales_arrived[ksub], 2 * sizeof(typename G::p3_scales_t));
                        wait(p3_scales_arrived[ksub], get_phasebit<0>(issue_phasebits, 6 + ksub));
                        #pragma unroll
                        for (int ii = 0; ii < C::P3_MMA_PER_TILE; ++ii) {
                            auto p3_B_sc_sub = p3_B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                            auto &B_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&p3_scales[ksub].B_sc.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(p3_B_sc_sub, B_sc_sm_sub);
                        }

                        tma::expect_bytes(p3_tiles_arrived[ksub], 2 * sizeof(typename G::p3_tiles_t));
                        wait(p3_tiles_arrived[ksub], get_phasebit<0>(issue_phasebits, 6 + ksub));
                        mm2_ABt(
                            p3_accum, fp4_staging[phase].G_row, p3_tiles[ksub].B,
                            p3_A_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_MMA_PER_TILE * 16>>(0),
                            p3_B_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_MMA_PER_TILE * 32>>(0),
                            p3_inputs_finished[ksub]);
                        tensor_commit<2>(p3_outputs_arrived[ksub]);
                        update_phasebit<0>(issue_phasebits, 6 + ksub);
                    }

                    wait(p3_outputs_arrived[ksub], get_phasebit<0>(acc_phasebits, 9 + ksub));
                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                        out_rt D_reg_fl;
                        warpgroup::load_async(
                            D_reg_fl,
                            p3_accum.template subtile<full_tt_fl<C::Nb_out / C::EPI_PIPE_DEPTH>>(0, epi * (C::Nb_out / C::EPI_PIPE_DEPTH)));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warp::mul(D_reg_fl, D_reg_fl, p3_scale);
                        if (vocab_block_idx == 0) warp::copy(D_acc[ksub][epi], D_reg_fl);
                        else                      warp::add(D_acc[ksub][epi], D_acc[ksub][epi], D_reg_fl);
                        warpgroup::sync(1);
                        tensor_after_thread_sync();
                    }

                    update_phasebit<0>(acc_phasebits, 9 + ksub);
                    if (ksub == last_valid_ksub) {
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                        update_phasebit<0>(acc_phasebits, 0);
                    }
                }

                if (warp::elect_leader() && cta_id == 0 && warpgroup::warpid() == 0) {
                    update_phasebit<0>(issue_phasebits, 5);
                    update_phasebit<1>(issue_phasebits, 0);
                }

                phase ^= 1;
            }

            #pragma unroll
            for (int ksub = 0; ksub < C::K_SUPERBLOCK; ++ksub) {
                const int k_block_idx = k_block_base + ksub;
                if (k_block_idx >= num_k_blocks) continue;

                rt_bf<C::Mb / 8, C::Nb_out / C::EPI_PIPE_DEPTH> D_reg_bf[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warp::copy(D_reg_bf[epi], D_acc[ksub][epi]);
                }

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg_bf[epi]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                        g.dE_out, output_tiles.D[epi % C::NUM_D_TILES],
                        {row_block_idx * 2 + cta_id, C::EPI_PIPE_DEPTH * k_block_idx + epi});
                }
            }
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    } else if (warpgroup_id == C::EPILOGUE_WARPGROUP_ID) {
        everyone::tma::cluster::wait_aligned();
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}];
        const int lane_id = threadIdx.x % 32;
        using logits_rt = rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        using logits_rt_bf = rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        constexpr float kFp4Max = 6.0f;
        constexpr float kE4M3Max = 448.0f;
        const float fp4_row_sg = fmaxf(g.grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
        const float fp4_row_senc = 1.0f / fp4_row_sg;

        uint32_t epi_phasebits = 0xFFFF0000;
        int phase = 0;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            const int supergroup_idx = block_idx / num_blocks_per_supergroup;
            const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            const int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;

            for (int vocab_block_idx = 0; vocab_block_idx < num_vocab_blocks; ++vocab_block_idx) {
                wait(outputs_arrived, get_phasebit<0>(epi_phasebits, 0));

                const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

                auto &phase1_accum = (phase == 0) ? out_tm_0 : out_tm_1;
                const uint32_t fp4_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging[phase].G_row.data[0]));
                uint8_t *fp4_sc_bytes = reinterpret_cast<uint8_t*>(&fp4_staging[phase].G_row_sc.data[0]);

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

                logits_rt D_pipe[2];
                logits_rt_bf D_bf;
                warpgroup::load_async(
                    D_pipe[0],
                    phase1_accum.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, 0));
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
                                phase1_accum.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, (epi + 1) * (C::Nb / C::EPI_PIPE_DEPTH)));
                        }
                    }

                    auto &D_fl = D_pipe[cur_slot];
                    warp::mul(D_fl, D_fl, global_scale);
                    const int col_start = vocab_block_idx * C::Nb + epi * (C::Nb / C::EPI_PIPE_DEPTH);

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
                        if (tgt_x >= col_start && tgt_x < col_start + (C::Nb / C::EPI_PIPE_DEPTH)) {
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
                        if (tgt_y >= col_start && tgt_y < col_start + (C::Nb / C::EPI_PIPE_DEPTH)) {
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

                    warp::mul(D_fl, D_fl, g.grad_scale);
                    warp::copy(D_bf, D_fl);
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[0], D_bf);
                    warpgroup::sync(1);

                    const int quant_row = threadIdx.x;
                    if (quant_row < C::Mb / 2) {
                        const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
                        #pragma unroll
                        for (int group16 = 0; group16 < (C::Nb / C::EPI_PIPE_DEPTH) / 16; ++group16) {
                            bf16_2 vals[8];
                            float amax = 0.0f;
                            #pragma unroll
                            for (int pair = 0; pair < 8; ++pair) {
                                const int col = group16 * 16 + pair * 2;
                                move<bf16_2>::lds(vals[pair], G::Out_tile::idx(d_base, {quant_row, col}));
                                amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                                amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                            }

                            const float row_scale = amax * (1.0f / kFp4Max);
                            const float row_rcp = (amax > 0.0f) ? (kFp4Max / amax) : 0.0f;

                            #pragma unroll
                            for (int pair = 0; pair < 8; ++pair) {
                                const uint8_t fp4_pair = quantize_fp4_pair_v5_dE(
                                    __bfloat162float(vals[pair].x),
                                    __bfloat162float(vals[pair].y),
                                    row_rcp);
                                const int fp4x2_col = (epi * (C::Nb / C::EPI_PIPE_DEPTH) + group16 * 16 + pair * 2) / 2;
                                const uint32_t fp4_addr = G::G_fp4_row_tile::idx(fp4_base, {quant_row, fp4x2_col});
                                asm volatile("{st.shared.b8 [%0], %1;}" :: "r"(fp4_addr), "r"((uint32_t)fp4_pair));
                            }

                            const __nv_fp8_e4m3 sc = __nv_fp8_e4m3(row_scale * fp4_row_senc);
                            const int global_col_16 = epi * (C::Nb / C::EPI_PIPE_DEPTH) + group16 * 16;
                            const int kgroup = global_col_16 / 64;
                            const int col_16_in_64 = (global_col_16 / 16) % 4;
                            const int sr = quant_row % 32;
                            const int rr = (quant_row / 32) % 4;
                            const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                            fp4_sc_bytes[kgroup * 512 + byte_idx] = *reinterpret_cast<const uint8_t*>(&sc);
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

                warpgroup::sync(1);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (warpgroup::warpid() == 0 && lane_id == 0) {
                    arrive(fp4_ready, 1);
                    if (cta_id == 0) {
                        arrive(fp4_ready_cluster, 1);
                    } else {
                        uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_ready_cluster));
                        uint32_t remote_addr;
                        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote_addr) : "r"(local_addr), "r"(0));
                        asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n" :: "r"(remote_addr), "r"((uint32_t)1) : "memory");
                    }
                }

                update_phasebit<0>(epi_phasebits, 0);
                phase ^= 1;
            }
        }
    }
}

} // namespace nvfp4_cce_backward_v5_dE_superk4_experimental
