#pragma once
// ================================================================
// NVFP4 CCE Backward v2 — Fused Softmax Gradient + FP4 Quantization
//
// Phase 1: FP4 GEMM recomputes logits = E_fp4 @ C_fp4^T
// Phase 2: Consumer computes G = grad_scale * (softmax(logits) - 1[target])
// Phase 3: Consumer stores G to global memory:
//   BF16 mode (USE_BF16_ACCUM=true):  Store G as BF16 via TMA
//   FP4  mode (USE_BF16_ACCUM=false): Quantize G to NVFP4 on-the-fly
//          (row-wise for dE = G @ C, col-wise for dC = G.T @ E)
//          Per-16-element FP8 micro scales, fixed global scale (bounded grad)
//
// After this kernel, separate GEMM calls compute dE/dC:
//   dE = G_output @ C_quantized   (FP4 or BF16 GEMM)
//   dC = G_output^T @ E_quantized (FP4 or BF16 GEMM)
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v2 {

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
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = 2;
    static constexpr bool USE_BF16_ACCUM = _USE_BF16_ACCUM;
};

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _USE_BF16_ACCUM = true, bool _PINGPONG = true>
struct experimental_config_epipipe {
    using base = config<_LOAD_PIPE_DEPTH, _SUPERGROUP_SIZE, _USE_BF16_ACCUM, _PINGPONG>;
    static constexpr int CLUSTER_SIZE = base::CLUSTER_SIZE;
    static constexpr bool USE_PDL = base::USE_PDL;
    static constexpr int CONSUMER_WARPGROUPS = base::CONSUMER_WARPGROUPS;
    static constexpr int PRODUCER_WARPGROUPS = base::PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPGROUPS = base::NUM_WARPGROUPS;
    static constexpr int NUM_WARPS = base::NUM_WARPS;
    static constexpr int NUM_THREADS = base::NUM_THREADS;
    static constexpr int LOAD_PIPE_DEPTH = base::LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = base::EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = true;
    static constexpr bool PINGPONG = base::PINGPONG;
    static constexpr int SUPERGROUP_SIZE = base::SUPERGROUP_SIZE;
    static constexpr int Mb = base::Mb;
    static constexpr int Nb = base::Nb;
    static constexpr int Kb = base::Kb;
    static constexpr int B_SC_SIZE = base::B_SC_SIZE;
    static constexpr int MMA_PER_TILE = base::MMA_PER_TILE;
    static constexpr int NUM_D_TILES = base::NUM_D_TILES;
    static constexpr bool USE_BF16_ACCUM = base::USE_BF16_ACCUM;
};

// =========================================================================
// FP4 quantization helpers
// =========================================================================

// FP4 E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
// Quantize float to nearest FP4 E2M1 (4-bit nibble: 1 sign + 2 exp + 1 mantissa)
__device__ __forceinline__ uint8_t float_to_fp4(float val) {
    float aval = fabsf(val);
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t enc;
    if      (aval < 0.25f)  enc = 0;       // 0
    else if (aval < 0.75f)  enc = 1;       // 0.5
    else if (aval < 1.25f)  enc = 2;       // 1.0
    else if (aval < 1.75f)  enc = 3;       // 1.5
    else if (aval < 2.5f)   enc = 4;       // 2.0
    else if (aval < 3.5f)   enc = 5;       // 3.0
    else if (aval < 5.0f)   enc = 6;       // 4.0
    else                    enc = 7;       // 6.0
    return sign | enc;
}

// Pack pair of values into fp4x2 byte: lo nibble = first, hi nibble = second
__device__ __forceinline__ uint8_t quantize_fp4_pair(float v0, float v1, float rcp_scale) {
    uint8_t q0 = float_to_fp4(v0 * rcp_scale);
    uint8_t q1 = float_to_fp4(v1 * rcp_scale);
    return q0 | (q1 << 4);
}

struct fp4_scale_contract {
    uint8_t scale_byte;
    float quant_rcp_scale;
};

__device__ __forceinline__ fp4_scale_contract make_fp4_scale_contract(
    float amax,
    float g_sg_rcp,
    float g_sg,
    bool encode_centric)
{
    constexpr float kFp4Max = 6.0f;
    constexpr float kE4M3Max = 448.0f;

    const float exact_scale = fmaxf(amax / kFp4Max, 1.0f / 65536.0f);
    const float exact_rcp_scale = 1.0f / exact_scale;

    fp4_scale_contract contract{};
    float stored_scale = 0.0f;
    if (encode_centric) {
        stored_scale = fminf(exact_rcp_scale * g_sg, kE4M3Max);
    } else {
        stored_scale = exact_scale * g_sg_rcp;
    }
    reinterpret_cast<__nv_fp8_e4m3&>(contract.scale_byte) = __nv_fp8_e4m3(stored_scale);
    const float rounded_stored_scale = float(reinterpret_cast<__nv_fp8_e4m3&>(contract.scale_byte));
    if (encode_centric) {
        contract.quant_rcp_scale = rounded_stored_scale * g_sg_rcp;
    } else {
        contract.quant_rcp_scale =
            (rounded_stored_scale > 0.0f) ? (1.0f / (rounded_stored_scale * g_sg)) : 0.0f;
    }
    return contract;
}

// =========================================================================
// Globals
// =========================================================================
template <typename C>
struct globals {
    // FP4 tiles for logit recomputation (A=E_fp4, B=C_fp4)
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    // BF16 output tile (for BF16 mode — same as v1)
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    // FP4 output tiles (for FP4 mode — row-quantized G)
    // Row-quantized G: each tile is (Mb/2, Nb/2) in fp4x2 layout
    using G_fp4_row_tile = st_fp4e2m1_2<C::Mb/2, C::Nb/2>;
    // Scale tile for row-quantized: per-16-element blocks
    // For row-quant of (128, 128): 128 rows, 128/16 = 8 scale blocks per row
    // = 128 * 8 = 1024 FP8 scales = st_fp8e4m3<128, 8>
    // But TK NVFP4 scale layout is [rows/128, cols/64, 512] fp8
    using G_sc_row_tile  = st_hf<4, 256, false>; // Same layout as input scales

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    // FP4 G output globals
    using G_fp4_row_gl   = gl<fp4e2m1_2,  1,  1, -1, -1, G_fp4_row_tile>;
    using G_sc_row_gl    = gl<half,       1, -1, -1, 256, G_sc_row_tile>;
    using G_sg_row_gl    = gl<float,      1,  1,  1,  1>;

    // FP4 inputs
    A_fp4x2_gl     A;         // E_fp4 (M_padded, K)
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;         // C_fp4 (V_padded, K)
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;

    // BF16 mode output
    D_gl           D_out;     // BF16 grad_logits (M, V)

    // FP4 mode outputs (row-quantized G for dE = G @ C)
    G_fp4_row_gl   G_fp4_row;
    G_sc_row_gl    G_sc_row;
    G_sg_row_gl    G_sg_row;

    // Backward inputs
    const float* lse;
    const int64_t* targets;
    float grad_scale;
    float filter_eps;
    int M;
    int N;    // V (vocab)
    bool encode_centric;

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
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t);
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};

// =========================================================================
// Main kernel
// =========================================================================
template <typename C>
__device__ inline void backward_kernel_v2(const globals<C>& g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        if constexpr (C::USE_BF16_ACCUM) {
            g.D_out.template prefetch_tma<typename G::D_tile>();
        }
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
                    tma::cluster::load_async(input_scales[stage].A, g.A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if (cta_id == 0) tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // ======================== MMA WARP ========================
            everyone::tma::cluster::wait();
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);

            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

            int phase = 0;

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
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                if (phase == 0) do_mma_block(out_tm_0);
                else            do_mma_block(out_tm_1);
                tensor_commit<2>(outputs_arrived);
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

        const float a_sg = g.A_sc_global[{0}];
        const float b_sg = g.B_sc_global[{0}];
        const float global_scale = a_sg * b_sg;
        const float g_sg = g.G_sg_row[{0}];
        const float g_sg_rcp = (g_sg > 0.0f) ? (1.0f / g_sg) : 0.0f;

        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
        using subtile_rt = rt_fl<C::Mb / 8, SUBTILE_COLS>;
        using subtile_rt_bf = rt_bf<C::Mb / 8, SUBTILE_COLS>;

        const int lane_id = threadIdx.x % 32;
        int phase = 0;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
            int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);

            // Precompute per-lane target and LSE
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

            auto compute_subtile = [&](subtile_rt& D_fl, int col_start) {
                warp::mul(D_fl, D_fl, global_scale);

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
            };

            auto quantize_fp4_subtile = [&](subtile_rt& D_fl, int col_start) {
                #pragma unroll
                for (int i = 0; i < subtile_rt::height; i++) {
                    int global_row_x = warp_row_base + i * 16 + lane_id / 4;
                    int global_row_y = warp_row_base + i * 16 + 8 + lane_id / 4;

                    #pragma unroll
                    for (int j = 0; j < subtile_rt::width; j++) {
                        float vals_x[4], vals_y[4];
                        vals_x[0] = D_fl.tiles[i][j].data[0].x;
                        vals_x[1] = D_fl.tiles[i][j].data[0].y;
                        vals_x[2] = D_fl.tiles[i][j].data[2].x;
                        vals_x[3] = D_fl.tiles[i][j].data[2].y;
                        vals_y[0] = D_fl.tiles[i][j].data[1].x;
                        vals_y[1] = D_fl.tiles[i][j].data[1].y;
                        vals_y[2] = D_fl.tiles[i][j].data[3].x;
                        vals_y[3] = D_fl.tiles[i][j].data[3].y;

                        float amax_x = 0.0f, amax_y = 0.0f;
                        #pragma unroll
                        for (int v = 0; v < 4; v++) {
                            amax_x = fmaxf(amax_x, fabsf(vals_x[v]));
                            amax_y = fmaxf(amax_y, fabsf(vals_y[v]));
                        }
                        #pragma unroll
                        for (int offset = 1; offset < 4; offset <<= 1) {
                            amax_x = fmaxf(amax_x, __shfl_xor_sync(0xFFFFFFFF, amax_x, offset));
                            amax_y = fmaxf(amax_y, __shfl_xor_sync(0xFFFFFFFF, amax_y, offset));
                        }

                        const fp4_scale_contract contract_x =
                            make_fp4_scale_contract(amax_x, g_sg_rcp, g_sg, g.encode_centric);
                        const fp4_scale_contract contract_y =
                            make_fp4_scale_contract(amax_y, g_sg_rcp, g_sg, g.encode_centric);

                        uint8_t fp4_x_lo = quantize_fp4_pair(vals_x[0], vals_x[1], contract_x.quant_rcp_scale);
                        uint8_t fp4_x_hi = quantize_fp4_pair(vals_x[2], vals_x[3], contract_x.quant_rcp_scale);
                        uint8_t fp4_y_lo = quantize_fp4_pair(vals_y[0], vals_y[1], contract_y.quant_rcp_scale);
                        uint8_t fp4_y_hi = quantize_fp4_pair(vals_y[2], vals_y[3], contract_y.quant_rcp_scale);

                        if (global_row_x < g.M) {
                            uint8_t* fp4_ptr = reinterpret_cast<uint8_t*>(
                                &g.G_fp4_row[{global_row_x / (C::Mb/2), col_start / (C::Nb / 2)}]);
                            int local_row = global_row_x % (C::Mb/2);
                            int local_col_lo = (lane_id % 4) * 2 + j * 16;
                            int local_col_hi = local_col_lo + 8;
                            fp4_ptr[local_row * (C::Nb/2) + local_col_lo/2] = fp4_x_lo;
                            fp4_ptr[local_row * (C::Nb/2) + local_col_hi/2] = fp4_x_hi;
                        }
                        if (global_row_y < g.M) {
                            uint8_t* fp4_ptr = reinterpret_cast<uint8_t*>(
                                &g.G_fp4_row[{global_row_y / (C::Mb/2), col_start / (C::Nb / 2)}]);
                            int local_row = global_row_y % (C::Mb/2);
                            int local_col_lo = (lane_id % 4) * 2 + j * 16;
                            int local_col_hi = local_col_lo + 8;
                            fp4_ptr[local_row * (C::Nb/2) + local_col_lo/2] = fp4_y_lo;
                            fp4_ptr[local_row * (C::Nb/2) + local_col_hi/2] = fp4_y_hi;
                        }

                        if ((lane_id % 4) == 0) {
                            int sc_col_block = (col_start + j * 16) / 16;
                            if (global_row_x < g.M) {
                                reinterpret_cast<uint8_t*>(
                                    &g.G_sc_row[{global_row_x / 128, sc_col_block, 0, 0}]
                                )[global_row_x % 128] = contract_x.scale_byte;
                            }
                            if (global_row_y < g.M) {
                                reinterpret_cast<uint8_t*>(
                                    &g.G_sc_row[{global_row_y / 128, sc_col_block, 0, 0}]
                                )[global_row_y % 128] = contract_y.scale_byte;
                            }
                        }
                    }
                }
            };

            auto run_serial_epilogue = [&](auto& accum) {
                subtile_rt D_regs_fl[C::EPI_PIPE_DEPTH];
                subtile_rt_bf D_regs_bf[C::EPI_PIPE_DEPTH];

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    warpgroup::load_async(D_regs_fl[epi], accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                }

                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                    int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                    compute_subtile(D_regs_fl[epi], col_start);
                    warp::copy(D_regs_bf[epi], D_regs_fl[epi]);
                }

                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

                bool tile_is_filtered = false;
                if (g.filter_eps > 0.0f) {
                    float local_max = 0.0f;
                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                        #pragma unroll
                        for (int i = 0; i < subtile_rt::height; i++) {
                            #pragma unroll
                            for (int j = 0; j < subtile_rt::width; j++) {
                                #pragma unroll
                                for (int k = 0; k < 4; k++) {
                                    local_max = fmaxf(local_max, fabsf(D_regs_fl[epi].tiles[i][j].data[k].x));
                                    local_max = fmaxf(local_max, fabsf(D_regs_fl[epi].tiles[i][j].data[k].y));
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
                    __shared__ float filter_max_smem[WARPGROUP_WARPS];
                    if (lane_id == 0) filter_max_smem[warpgroup::warpid()] = local_max;
                    warpgroup::sync(1);
                    float global_max = 0.0f;
                    for (int w = 0; w < WARPGROUP_WARPS; w++)
                        global_max = fmaxf(global_max, filter_max_smem[w]);
                    tile_is_filtered = (global_max < g.filter_eps);
                }

                if (!tile_is_filtered) {
                    if constexpr (C::USE_BF16_ACCUM) {
                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                            warpgroup::sync(1);
                            warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_regs_bf[epi]);
                            warpgroup::sync(1);
                            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                                g.D_out, output_tiles.D[epi % C::NUM_D_TILES],
                                {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + epi});
                        }
                    } else {
                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; epi++) {
                            int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;
                            quantize_fp4_subtile(D_regs_fl[epi], col_start);
                        }
                    }
                }
            };

            auto run_overlap_epilogue = [&](auto& accum) {
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    subtile_rt D_fl;
                    int col_start = col_block_idx * C::Nb + epi * SUBTILE_COLS;

                    warpgroup::load_async(D_fl, accum.template subtile<full_tt_fl<SUBTILE_COLS>>(0, SUBTILE_COLS * epi));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);

                    compute_subtile(D_fl, col_start);

                    if (epi == C::EPI_PIPE_DEPTH - 1) {
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    }

                    if constexpr (C::USE_BF16_ACCUM) {
                        subtile_rt_bf D_bf;
                        warp::copy(D_bf, D_fl);
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_bf);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                            g.D_out, output_tiles.D[epi % C::NUM_D_TILES],
                            {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + epi});
                    } else {
                        quantize_fp4_subtile(D_fl, col_start);
                    }
                }
            };

            auto& accum = (phase == 0) ? out_tm_0 : out_tm_1;
            if constexpr (C::OVERLAP_EPI) {
                if (g.filter_eps > 0.0f) run_serial_epilogue(accum);
                else                     run_overlap_epilogue(accum);
            } else {
                run_serial_epilogue(accum);
            }

            update_phasebit<0>(phasebits, 0);
            phase ^= 1;
        }
        warpgroup::sync(1);
        if constexpr (C::USE_BF16_ACCUM) {
            warpgroup::tma::store_async_read_wait<0>();
        }
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace nvfp4_cce_backward_v2
