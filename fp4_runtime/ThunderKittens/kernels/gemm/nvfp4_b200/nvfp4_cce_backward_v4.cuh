#pragma once
// ================================================================
// NVFP4 CCE Backward v4 — Full One-Pass Fusion (Algorithm 3)
//
// Single kernel that fuses:
//   Phase 1: FP4 GEMM recomputes logits A = E_fp4 @ C_fp4^T
//            Softmax gradient G = grad_scale * (exp(A - LSE) - 1[target])
//   Phase 2: Row-quantize G to FP4 in SMEM (G never touches HBM!)
//            Also output BF16 G via TMA for separate dC GEMM
//   Phase 3: Inner K-loop GMMA: dE += G_fp4_smem @ C_col_fp4
//            dE is atomically accumulated to global memory
//
// Architecture: MMA warp does Phase 1 GEMM, then Phase 3 GEMM(s)
//               Producer warps load Phase 1 tiles, then Phase 3 tiles
//               Consumer processes both phases' outputs
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v4 {

// =========================================================================
// Config
// =========================================================================
template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _USE_BF16_ACCUM = true, bool _PINGPONG = true>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = true;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = 4;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;   // rows per block
    static constexpr int Nb = 128;   // cols per block (V dim = Phase 3 reduction dim)
    static constexpr int Kb = 256;   // Phase 1 reduction dim

    // Phase 3 dE GEMM dimensions
    // dE[Mb/2, Nb_out] = G_fp4[Mb/2, Nb] @ C_col[Nb_out, Nb]^T
    // Nb_out = output columns per Phase 3 chunk (K dimension of dE)
    static constexpr int Nb_out = 128;

    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;       // Phase 1: 4 MMA slices
    static constexpr int P3_MMA_PER_TILE = Nb/64;    // Phase 3: 2 MMA slices (Nb=128 reduction)

    static constexpr int NUM_D_TILES = 2;
    static constexpr bool USE_BF16_ACCUM = _USE_BF16_ACCUM;
};

// =========================================================================
// FP4 quantization helpers
// =========================================================================
__device__ __forceinline__ uint8_t float_to_fp4(float val) {
    float aval = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t enc;
    if      (aval < 0.25f)  enc = 0;
    else if (aval < 0.75f)  enc = 1;
    else if (aval < 1.25f)  enc = 2;
    else if (aval < 1.75f)  enc = 3;
    else if (aval < 2.5f)   enc = 4;
    else if (aval < 3.5f)   enc = 5;
    else if (aval < 5.0f)   enc = 6;
    else                    enc = 7;
    return sign | enc;
}

__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float rcp_scale) {
    uint8_t q0 = float_to_fp4(v0 * rcp_scale);
    uint8_t q1 = float_to_fp4(v1 * rcp_scale);
    return q0 | (q1 << 4);
}

// =========================================================================
// Globals
// =========================================================================
template <typename C>
struct globals {
    // Phase 1: FP4 tiles for logit recomputation
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;  // E_fp4    [128, 128]
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;   // C_fp4    [64, 128]
    using B_sc_tile    = st_hf<4, 256, false>;

    // Phase 3: C_col (column-quantized C for dE GEMM)
    // B operand: C_col[Nb_out/2, Nb/2] as fp4x2 tile = [64, 64] per CTA
    using P3_B_fp4x2_tile = st_fp4e2m1_2<C::Nb_out/2, C::Nb/2>;
    using P3_B_sc_tile    = st_hf<2, 256, false>;  // 2 x 512B scale groups for Nb=128

    // Phase 2: G_fp4 staging in SMEM (row-quantized softmax grad)
    // This is the A operand for Phase 3 GMMA
    using G_fp4_row_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;  // [128, 64] = 8KB
    using G_sc_row_tile  = st_hf<2, 256, false>;              // 1KB scale tile for Nb=128

    // BF16 output: grad_logits for dC GEMM
    using D_tile = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    // dE output tile (BF16)
    using dE_tile = st_bf<C::Mb/2, C::Nb_out>;  // [128, 128]

    // ═══ TMA globals ═══
    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    // Phase 3 tile globals
    using P3_B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, P3_B_fp4x2_tile>;
    using P3_B_sc_gl        = gl<half,       1, -1, -1, 256, P3_B_sc_tile>;
    using P3_B_sc_global_gl = gl<float,      1,  1,  1,  1>;

    // dE output global
    using dE_gl = gl<bf16, 1, 1, -1, -1, dE_tile>;

    // For unused FP4 output (compat fields)
    using G_fp4_row_gl = gl<fp4e2m1_2,  1,  1, -1, -1, G_fp4_row_tile>;
    using G_sg_row_gl  = gl<float,      1,  1,  1,  1>;

    // ═══ Inputs ═══
    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;

    // Phase 3: C column-quantized
    P3_B_fp4x2_gl      C_col;
    P3_B_sc_gl          C_col_sc;
    P3_B_sc_global_gl   C_col_sc_global;

    // ═══ Outputs ═══
    D_gl           D_out;     // BF16 grad_logits (M, V)
    dE_gl          dE_out;    // BF16 dE output (M, K) — atomically accumulated

    // Compat fields (unused in v4)
    G_fp4_row_gl   G_fp4_row;
    uint8_t*       G_sc_row_ptr;
    int            G_sc_row_kgroups;
    G_sg_row_gl    G_sg_row;
    uint8_t*       G_fp4_col_ptr;
    uint8_t*       G_sc_col_ptr;
    int            G_sc_col_kgroups;

    // Backward inputs
    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;     // V dimension
    int K;     // embedding dimension (for Phase 3 K-loop)
    bool encode_centric;

    // Phase 1 SMEM structures
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

    // Phase 3 SMEM structures (overlaid on Phase 1 input buffers after Phase 1 completes)
    struct p3_tiles_t {
        P3_B_fp4x2_tile B;    // C_col tile [64, 64]
    };
    struct p3_scales_t {
        P3_B_sc_tile B_sc;    // C_col scales
    };

    // Phase 2 SMEM staging for FP4 G
    struct fp4_staging_t {
        G_fp4_row_tile G_row;     // [128, 64] fp4x2 = 8KB
        G_sc_row_tile  G_row_sc;  // scales for G (1KB)
    };

    __host__ inline dim3 grid() const {
        int padded_M = A.rows();
        int padded_N = B.rows();
        int num_row_blocks = padded_M / C::Mb;
        int num_col_blocks = padded_N / C::Nb;
        int total = num_row_blocks * num_col_blocks;
        int max_blocks = num_sms();
        int grid_size = min(total, max_blocks);
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(grid_size);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        // Phase 1 SMEM
        constexpr int phase1_smem = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(outputs_t);
        // Phase 2/3 additional SMEM (FP4 staging + P3 tile/scale buffers)
        constexpr int fp4_staging_smem = sizeof(fp4_staging_t);
        constexpr int p3_smem = sizeof(p3_tiles_t) + 1024 + sizeof(p3_scales_t) + 1024;
        constexpr int _dynamic_shared_memory = phase1_smem + fp4_staging_smem + p3_smem;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

// =========================================================================
// Main kernel — v4 fused backward + dE GEMM
// =========================================================================
template <typename C>
__device__ inline void backward_kernel_v4(const globals<C>& g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D_out.template prefetch_tma<typename G::D_tile>();
        g.C_col.template prefetch_tma<typename G::P3_B_fp4x2_tile>();
        g.C_col_sc.template prefetch_tma<typename G::P3_B_sc_tile>();
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
    const int num_k_chunks = g.K / C::Nb_out;  // Phase 3 K-loop iterations
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    // Phase 1 SMEM
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();
    // Phase 2 SMEM: FP4 staging
    typename G::fp4_staging_t   &fp4_staging                       = sm_allocator.allocate<G::fp4_staging_t>();
    // Phase 3 SMEM: C_col tile + scales (single pipeline depth)
    typename G::p3_tiles_t      &p3_tiles                          = sm_allocator.allocate<G::p3_tiles_t>();
    typename G::p3_scales_t     &p3_scales                         = sm_allocator.allocate<G::p3_scales_t>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    // Phase 3 sync
    __shared__ semaphore p3_tiles_arrived;
    __shared__ semaphore p3_scales_arrived;
    __shared__ semaphore p3_inputs_finished;
    __shared__ semaphore p3_outputs_arrived;
    __shared__ semaphore p3_outputs_finished;
    __shared__ semaphore fp4_ready;          // consumer→producer: local CTA G staging is ready
    __shared__ semaphore fp4_ready_cluster;  // both CTAs → MMA: cluster G staging is ready
    __shared__ float fp4_row_amax[WARPGROUP_WARPS];
    __shared__ float fp4_row_sg;

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
        init_semaphore(p3_tiles_arrived, 0, 1);
        init_semaphore(p3_scales_arrived, 0, 1);
        init_semaphore(p3_inputs_finished, 0, 1);
        init_semaphore(p3_outputs_arrived, 0, 1);
        init_semaphore(p3_outputs_finished, 0, C::CLUSTER_SIZE);
        init_semaphore(fp4_ready, 1, 0);
        init_semaphore(fp4_ready_cluster, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();

    // ======================== PRODUCER ========================
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // Load input tiles: Phase 1 + Phase 3
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                // ─── Phase 1: Load E_fp4, C_fp4 tiles for logit recompute ───
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }

                // ─── Phase 3: Load C_col tiles for dE GEMM ───
                if (num_k_chunks > 0) {
                    wait(fp4_ready, get_phasebit<1>(phasebits, 5));
                    for (int k = 0; k < num_k_chunks; k++) {
                        wait(p3_inputs_finished, get_phasebit<1>(phasebits, 6));
                        tma::cluster::load_async(p3_tiles.B, g.C_col, {k*2 + cta_id, col_block_idx}, p3_tiles_arrived, (uint16_t)(1<<cta_id), 0);
                        update_phasebit<1>(phasebits, 6);
                    }
                    update_phasebit<1>(phasebits, 5);
                }
            }
        } else if (warp_id == 2) {
            // Load input scales: Phase 1 + Phase 3
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                // ─── Phase 1: Load scales ───
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if (cta_id == 0) tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }

                // ─── Phase 3: Load C_col scales for dE GEMM ───
                if (num_k_chunks > 0) {
                    wait(fp4_ready, get_phasebit<1>(phasebits, 5));
                    for (int k = 0; k < num_k_chunks; k++) {
                        wait(p3_inputs_finished, get_phasebit<1>(phasebits, 6));
                        tma::cluster::load_async(p3_scales.B_sc, g.C_col_sc, {k, col_block_idx, 0}, p3_scales_arrived, (uint16_t)(1<<cta_id), 0);
                        update_phasebit<1>(phasebits, 6);
                    }
                    update_phasebit<1>(phasebits, 5);
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // ======================== MMA WARP ========================
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            // TMEM for Phase 1
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

            // TMEM for Phase 3 (reuse out_tm_0 for accumulator, new scale areas)
            // Phase 3 uses Nb_out output columns, same as Nb=128
            auto p3_out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb_out>>(0); // alias to out_tm_0!
            // Phase 3 scales: G scales (A for P3) and C_col scales (B for P3)
            // Reuse Phase 1's TMEM scale region (Phase 1 is done before Phase 3 starts)
            constexpr int p3_sc_offset = 256; // same as Phase 1's A_sc_tm start
            auto p3_A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::P3_MMA_PER_TILE>>(p3_sc_offset);
            auto p3_B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::P3_MMA_PER_TILE>>(p3_sc_offset + 4*C::P3_MMA_PER_TILE);

            int phase = 0;

            // Phase 1 GEMM block
            auto do_mma_block = [&](auto& accum) {
                for (int i = 0; i < num_iters_per_block; i++) {
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
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    if (i == 0) mm2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    else       mma2_ABt(accum, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                        inputs_finished[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            };

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                // ─── Phase 1: Logit GEMM ───
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                if (phase == 0) do_mma_block(out_tm_0);
                else            do_mma_block(out_tm_1);
                tensor_commit<2>(outputs_arrived);


                // ─── Phase 3: dE GEMM ───
                if (num_k_chunks > 0) {
                    wait(fp4_ready_cluster, get_phasebit<0>(phasebits, 5));
                    tensor_after_thread_sync();

                    // Load G scales to TMEM (A scales for Phase 3)
                    #pragma unroll
                    for (int ii = 0; ii < C::P3_MMA_PER_TILE; ii++) {
                        auto p3_A_sc_sub = p3_A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                        auto &G_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&fp4_staging.G_row_sc.data[0])+16*32*ii);
                        load_mxnv_scale_async2(p3_A_sc_sub, G_sc_sm_sub);
                    }

                    for (int k = 0; k < num_k_chunks; k++) {
                        if (k > 0) {
                            // The Phase 3 accumulator aliases cluster TMEM, so after the previous
                            // drain completes we need the same reuse fence as the outer Phase 1 ring.
                            tensor_after_thread_sync();
                        }
                        // Wait for C_col scales from TMA
                        tma::expect_bytes(p3_scales_arrived, 2*sizeof(G::p3_scales_t));
                        wait(p3_scales_arrived, get_phasebit<0>(phasebits, 6));

                        // Load C_col scales to TMEM (B scales for Phase 3)
                        #pragma unroll
                        for (int ii = 0; ii < C::P3_MMA_PER_TILE; ii++) {
                            auto p3_B_sc_sub = p3_B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                            auto &Ccol_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&p3_scales.B_sc.data[0])+16*32*ii);
                            load_mxnv_scale_async2(p3_B_sc_sub, Ccol_sc_sm_sub);
                        }

                        // Wait for C_col FP4 tiles from TMA
                        tma::expect_bytes(p3_tiles_arrived, 2*sizeof(G::p3_tiles_t));
                        wait(p3_tiles_arrived, get_phasebit<0>(phasebits, 6));

                        // Each k chunk is a distinct dE output tile, so use a fresh TMEM accumulator.
                        mm2_ABt(p3_out_tm, fp4_staging.G_row, p3_tiles.B,
                                p3_A_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_MMA_PER_TILE*16>>(0),
                                p3_B_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_MMA_PER_TILE*32>>(0),
                                p3_inputs_finished);

                        tensor_commit<2>(p3_outputs_arrived);

                        if (k + 1 < num_k_chunks) {
                            wait(p3_outputs_finished, get_phasebit<0>(phasebits, 7));
                        }
                        update_phasebit<0>(phasebits, 6);
                        update_phasebit<0>(phasebits, 7);
                    }

                    update_phasebit<0>(phasebits, 5);
                }
                update_phasebit<1>(phasebits, 0);
                phase ^= 1;
            }
        }
    // ======================== CONSUMER ========================
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
        auto p3_out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb_out>>(0); // alias

        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using subtile_rt_bf = rt_bf<C::Mb / 8, SUBTILE_COLS>;

        // Phase 3 output registers
        using p3_rt = rt_fl<C::Mb / 8, C::Nb_out>;

        const int lane_id = threadIdx.x % 32;
        int phase = 0;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // ════════════════════════════════════════════════════
            // Phase 1: Read logit GEMM result → softmax gradient G
            // ════════════════════════════════════════════════════
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

            subtile_rt D_regs_fl[C::EPI_PIPE_DEPTH];
            subtile_rt_bf D_regs_bf[C::EPI_PIPE_DEPTH];

            auto load_from_accum = [&](auto& accum) {
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    warpgroup::load_async(D_regs_fl[epi], accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                }
            };
            if (phase == 0) load_from_accum(out_tm_0);
            else            load_from_accum(out_tm_1);

            // Precompute targets and LSE
            int my_targets_x[subtile_rt::height];
            int my_targets_y[subtile_rt::height];
            float my_lse_x[subtile_rt::height];
            float my_lse_y[subtile_rt::height];
            #pragma unroll
            for (int i = 0; i < subtile_rt::height; i++) {
                int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                my_targets_x[i] = (global_row_x < g.M) ? (int)g.targets[global_row_x] : -1;
                my_targets_y[i] = (global_row_y < g.M) ? (int)g.targets[global_row_y] : -1;
                my_lse_x[i] = (global_row_x < g.M) ? g.lse[global_row_x] : INFINITY;
                my_lse_y[i] = (global_row_y < g.M) ? g.lse[global_row_y] : INFINITY;
            }

            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);

            // Compute G = grad_scale * (exp(logits - LSE) - 1[target])
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                subtile_rt& D_fl = D_regs_fl[epi];
                warp::mul(D_fl, D_fl, global_scale);
                int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;

                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    float lse_x = my_lse_x[i];
                    float lse_y = my_lse_y[i];
                    #pragma unroll
                    for (int j = 0; j < subtile_rt::width; j++) {
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            float lse_val = (k % 2 == 0) ? lse_x : lse_y;
                            D_fl.tiles[i][j].data[k].x = __expf(D_fl.tiles[i][j].data[k].x - lse_val);
                            D_fl.tiles[i][j].data[k].y = __expf(D_fl.tiles[i][j].data[k].y - lse_val);
                        }
                    }
                }
                // Subtract 1 at target
                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    int tgt_x = my_targets_x[i];
                    if (tgt_x >= col_start && tgt_x < col_start + SUBTILE_COLS) {
                        int local_col = tgt_x - col_start;
                        int j_idx = local_col / 16;
                        int within_tile = local_col % 16;
                        int k_half = within_tile / 8;
                        int pair_pos = (within_tile % 8) / 2;
                        if ((lane_id % 4) == pair_pos) {
                            int k_idx = k_half * 2;
                            if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                            else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
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
                            if ((local_col & 1) == 0) D_fl.tiles[i][j_idx].data[k_idx].x -= 1.0f;
                            else                      D_fl.tiles[i][j_idx].data[k_idx].y -= 1.0f;
                        }
                    }
                }
                // Zero out-of-bounds rows
                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                    int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;
                    #pragma unroll
                    for (int j = 0; j < subtile_rt::width; j++) {
                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            if (k % 2 == 0 && global_row_x >= g.M) {
                                D_fl.tiles[i][j].data[k].x = 0.0f;
                                D_fl.tiles[i][j].data[k].y = 0.0f;
                            }
                            if (k % 2 == 1 && global_row_y >= g.M) {
                                D_fl.tiles[i][j].data[k].x = 0.0f;
                                D_fl.tiles[i][j].data[k].y = 0.0f;
                            }
                        }
                    }
                }
                warp::mul(D_fl, D_fl, g.grad_scale);
                warp::copy(D_regs_bf[epi], D_fl);
            }

            // Free Phase 1 TMEM — MMA warp can start next block's Phase 1
            // (But first, MMA warp needs Phase 3, so we signal outputs_finished AFTER Phase 3)

            // ════════════════════════════════════════════════════
            // Phase 2a/2b setup: compute one tile-global FP4 scale
            // ════════════════════════════════════════════════════
            const int warp_id = warpgroup::warpid();
            {
                float local_tile_amax = 0.0f;

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    subtile_rt& D_fl = D_regs_fl[epi];
                    #pragma unroll
                    for (int i = 0; i < subtile_rt::height; i++) {
                        #pragma unroll
                        for (int j = 0; j < subtile_rt::width; j++) {
                            #pragma unroll
                            for (int kk = 0; kk < 4; kk++) {
                                local_tile_amax = fmaxf(local_tile_amax, fabsf(D_fl.tiles[i][j].data[kk].x));
                                local_tile_amax = fmaxf(local_tile_amax, fabsf(D_fl.tiles[i][j].data[kk].y));
                            }
                        }
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    local_tile_amax = fmaxf(local_tile_amax, __shfl_xor_sync(0xFFFFFFFF, local_tile_amax, offset));
                }
                if (lane_id == 0) fp4_row_amax[warp_id] = local_tile_amax;
                warpgroup::sync(1);

                if (warp_id == 0) {
                    float tile_amax = (lane_id < WARPGROUP_WARPS) ? fp4_row_amax[lane_id] : 0.0f;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        tile_amax = fmaxf(tile_amax, __shfl_xor_sync(0xFFFFFFFF, tile_amax, offset));
                    }
                    if (lane_id == 0) fp4_row_sg = (tile_amax > 0.0f) ? (tile_amax / 2688.0f) : 1.0f;
                }
                warpgroup::sync(1);
            }
            const float fp4_row_senc = 1.0f / fp4_row_sg;

            // ════════════════════════════════════════════════════
            // Phase 2a: Store BF16 G output via TMA (for dC GEMM)
            // Phase 2b: Quantize each canonical BF16 slice from shared memory
            // ════════════════════════════════════════════════════
            const uint32_t fp4_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.G_row.data[0]));
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_regs_bf[epi]);
                warpgroup::sync(1);

                // Quantize the canonical BF16 slice from shared memory rather than
                // depending on the MMA register packing inherited from v3.
                const int quant_row = threadIdx.x;
                if (quant_row < C::Mb / 2) {
                    const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[epi % C::NUM_D_TILES].data[0]));
                    #pragma unroll
                    for (int group16 = 0; group16 < SUBTILE_COLS / 16; group16++) {
                        bf16_2 vals[8];
                        float amax = 0.0f;
                        #pragma unroll
                        for (int pair = 0; pair < 8; pair++) {
                            const int col = group16 * 16 + pair * 2;
                            move<bf16_2>::lds(vals[pair], G::D_tile::idx(d_base, {quant_row, col}));
                            amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].x)));
                            amax = fmaxf(amax, fabsf(__bfloat162float(vals[pair].y)));
                        }

                        const float FP4_MAX = 6.0f;
                        const float FP8_MAX = 448.0f;
                        const float S_mult = (amax <= 1.0e-9f)
                            ? FP8_MAX
                            : fminf(FP4_MAX / (amax * fp4_row_senc), FP8_MAX);
                        const __nv_fp8_e4m3 S_mult_fp8 = __nv_fp8_e4m3(S_mult);
                        const float coeff = static_cast<float>(S_mult_fp8) * fp4_row_senc;

                        #pragma unroll
                        for (int pair = 0; pair < 8; pair++) {
                            const float2 scaled = {
                                __bfloat162float(vals[pair].x) * coeff,
                                __bfloat162float(vals[pair].y) * coeff
                            };
                            const uint8_t fp4_pair = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
                                scaled, __NV_E2M1, cudaRoundNearest));
                            const int fp4x2_col = (epi * SUBTILE_COLS + group16 * 16 + pair * 2) / 2;
                            const uint32_t fp4_addr = G::G_fp4_row_tile::idx(fp4_base, {quant_row, fp4x2_col});
                            asm volatile("{st.shared.b8 [%0], %1;}" :: "r"(fp4_addr), "r"((uint32_t)fp4_pair));
                        }

                        const float S_mult_val = static_cast<float>(S_mult_fp8);
                        __nv_fp8_e4m3 sc = __nv_fp8_e4m3(1.0f / S_mult_val);
                        const int global_col_16 = epi * SUBTILE_COLS + group16 * 16;
                        const int kgroup = global_col_16 / 64;
                        const int col_16_in_64 = (global_col_16 / 16) % 4;
                        const int sr = quant_row % 32;
                        const int rr = (quant_row / 32) % 4;
                        const int byte_idx = sr * 16 + rr * 4 + col_16_in_64;
                        reinterpret_cast<uint8_t*>(&fp4_staging.G_row_sc.data[0])[kgroup * 512 + byte_idx] =
                            *reinterpret_cast<uint8_t*>(&sc);
                    }
                }

                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    g.D_out, output_tiles.D[epi % C::NUM_D_TILES],
                    {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + epi});
            }
            warpgroup::sync(1);
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            if (num_k_chunks > 0 && warpgroup::warpid() == 0 && lane_id == 0) {
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

            // Drain each Phase 3 TMEM tile and atomically accumulate it into global dE.
            if (num_k_chunks > 0) {
                const float p3_global_scale = fp4_row_sg * g.C_col_sc_global[{0}];
                for (int k = 0; k < num_k_chunks; k++) {
                    wait(p3_outputs_arrived, get_phasebit<0>(phasebits, 8));

                    p3_rt dE_fl;
                    warpgroup::load_async(dE_fl, p3_out_tm);
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);

                    warp::mul(dE_fl, dE_fl, p3_global_scale);

                    const int wid = warpgroup::warpid();
                    const int warp_row_base_p3 = wid * (C::Mb / 8);

                    __nv_bfloat16* dE_ptr = reinterpret_cast<__nv_bfloat16*>(g.dE_out.raw_ptr);
                    #pragma unroll
                    for (int i = 0; i < p3_rt::height; i++) {
                        const int row_x = tile_row_base + warp_row_base_p3 + i * 16 + lane_id / 4;
                        const int row_y = tile_row_base + warp_row_base_p3 + i * 16 + 8 + lane_id / 4;
                        #pragma unroll
                        for (int j = 0; j < p3_rt::width; j++) {
                            const int col_base = k * C::Nb_out + j * 16 + (lane_id % 4) * 2;
                            const float2 d0 = dE_fl.tiles[i][j].data[0];
                            const float2 d1 = dE_fl.tiles[i][j].data[1];
                            const float2 d2 = dE_fl.tiles[i][j].data[2];
                            const float2 d3 = dE_fl.tiles[i][j].data[3];

                            if (row_x < g.M) {
                                const size_t off = static_cast<size_t>(row_x) * g.K + col_base;
                                if (col_base + 1 < g.K) {
                                    atomicAdd(reinterpret_cast<__nv_bfloat162*>(&dE_ptr[off]),
                                              __floats2bfloat162_rn(d0.x, d0.y));
                                }
                                if (col_base + 9 < g.K) {
                                    atomicAdd(reinterpret_cast<__nv_bfloat162*>(&dE_ptr[off + 8]),
                                              __floats2bfloat162_rn(d2.x, d2.y));
                                }
                            }
                            if (row_y < g.M) {
                                const size_t off = static_cast<size_t>(row_y) * g.K + col_base;
                                if (col_base + 1 < g.K) {
                                    atomicAdd(reinterpret_cast<__nv_bfloat162*>(&dE_ptr[off]),
                                              __floats2bfloat162_rn(d1.x, d1.y));
                                }
                                if (col_base + 9 < g.K) {
                                    atomicAdd(reinterpret_cast<__nv_bfloat162*>(&dE_ptr[off + 8]),
                                              __floats2bfloat162_rn(d3.x, d3.y));
                                }
                            }
                        }
                    }

                    warpgroup::tma::cluster::arrive(p3_outputs_finished, 0, 1);
                    update_phasebit<0>(phasebits, 8);
                }
            }

            // Now signal outputs_finished so MMA warp can start next block
            warpgroup::sync(1);
            tensor_after_thread_sync();
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
            update_phasebit<0>(phasebits, 0);
            phase ^= 1;
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace nvfp4_cce_backward_v4
