#pragma once
// ================================================================
// NVFP4 CCE Backward v5 dC experimental sandbox
//
// Grid ownership: (vocab_superblock, k_superblock)
// One vocab superblock = 2 adjacent 128-row vocab blocks.
//
// For each row-block we:
//  1. recompute/quantize the low vocab block into CTA0-local staged G^T
//  2. recompute/quantize the high vocab block into CTA1-local staged G^T
//  3. reuse the staged Gt_super across 2 adjacent k blocks:
//       dC_super[k0] += Gt_super @ E_col[k0]^T
//       dC_super[k1] += Gt_super @ E_col[k1]^T
//
// This deliberately avoids the toxic M=128 half_tt phase-3 contract.
//
// Experimental warpgroup split:
//  - producer WG: TMA only
//  - math WG: phase 1 / phase 3 tensor-core issue + dC accumulation
//  - epilogue WG: softmax + G^T FP4 staging
// ================================================================

#include "nvfp4_cce.cuh"

namespace nvfp4_cce_backward_v5_dC_superk4_experimental {

__device__ __forceinline__ uint8_t quantize_fp4_pair_v5(float v0, float v1, float rcp_scale) {
    const float2 scaled = {v0 * rcp_scale, v1 * rcp_scale};
    return static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
}

__device__ __forceinline__ void store_b8_cluster_v5(uint32_t local_addr, int target_cta, uint8_t value) {
    uint32_t addr = local_addr;
    if (target_cta != cluster_ctarank()) {
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                     : "=r"(addr)
                     : "r"(local_addr), "r"(target_cta));
    }
    asm volatile("{st.shared::cluster.b8 [%0], %1;}" :: "r"(addr), "r"((uint32_t)value));
}

__device__ __forceinline__ void store_b8_local_v5(uint32_t local_addr, uint8_t value) {
    asm volatile("{st.shared.b8 [%0], %1;}" :: "r"(local_addr), "r"((uint32_t)value));
}

__device__ __forceinline__ void arrive_remote_cluster_v5(semaphore &sem, int target_cta, uint32_t count = 1) {
    uint32_t local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(remote_addr)
                 : "r"(local_addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
                 :: "r"(remote_addr), "r"(count)
                 : "memory");
}

template <int _LOAD_PIPE_DEPTH, int _SUPERGROUP_SIZE, bool _PINGPONG = true>
struct config {
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_SUPERGROUP_SIZE > 0);

    static constexpr int CLUSTER_SIZE = 2;
    static constexpr bool USE_PDL = false;

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
    static constexpr int EPI_PIPE_DEPTH = 4;
    static constexpr bool OVERLAP_EPI = false;
    static constexpr bool PINGPONG = _PINGPONG;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = 256;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int Nb_out = 128;

    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int P3_MMA_PER_TILE = Nb / 64;
    static constexpr int P3_SCALE_CHUNKS = Mb / 64;
    static constexpr int P3_A_SCALE_CHUNKS = Mb / 64;
    static constexpr int NUM_D_TILES = 2;
};

static constexpr int DEBUG_TRACE_WIDTH = 9;
enum debug_trace_role : int {
    TRACE_ROLE_MATH = 1,
    TRACE_ROLE_EPILOGUE = 2,
};
enum debug_trace_marker : int {
    TRACE_BLOCK_ENTRY = 1,
    TRACE_ROW_ENTRY = 2,
    TRACE_PRE_OUTPUTS_FINISHED = 3,
    TRACE_POST_OUTPUTS_FINISHED = 4,
    TRACE_ROW_COMPLETE = 5,
    TRACE_BLOCK_TAIL_OUTPUTS_FINISHED = 6,
    TRACE_FINAL_STORE_ENTRY = 7,
    TRACE_FINAL_STORE_EXIT = 8,
    TRACE_EPI_PRE_ROW_SYNC = 9,
    TRACE_EPI_POST_ROW_SYNC = 10,
    TRACE_EPI_PRE_CLEAR_SYNC = 11,
    TRACE_EPI_POST_CLEAR_SYNC = 12,
    TRACE_EPI_PRE_CLEAR_WAIT = 13,
    TRACE_EPI_POST_CLEAR_WAIT = 14,
    TRACE_EPI_PRE_OUTPUTS_ARRIVED = 15,
    TRACE_EPI_POST_OUTPUTS_ARRIVED = 16,
    TRACE_EPI_PRE_GT_ROW_CLEAR = 17,
    TRACE_EPI_POST_GT_ROW_CLEAR = 18,
    TRACE_EPI_PRE_GT_SC_CLEAR = 19,
    TRACE_EPI_POST_GT_SC_CLEAR = 20,
    TRACE_EPI_PRE_SCRATCH_WAIT = 21,
    TRACE_EPI_POST_SCRATCH_WAIT = 22,
};

template <typename C>
struct debug_globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    using P3_B_fp4x2_tile = st_fp4e2m1_2<C::Nb_out/2, C::Mb/2>;
    using P3_B_sc_tile    = st_hf<4, 256, false>;

    // Full 128-row local G^T view per CTA for the 256-row superblock phase-3 path.
    using Gt_fp4_row_tile = st_fp4e2m1_2<C::Nb, C::Mb/2>;
    // Match the generic NVFP4 GEMM scale staging contract exactly.
    // The producer later reinterprets this as 4 contiguous 32x16 fp8 chunks
    // for load_mxnv_scale_async2, just like the working generic path.
    using Gt_sc_row_tile  = st_hf<4, 256, false>;

    using Out_tile      = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH>;
    using Out_sm_tile   = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;
    using Debug_raw_tile = st_fl<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;

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

    P3_B_fp4x2_gl     E_col;
    P3_B_sc_gl        E_col_sc;
    P3_B_sc_global_gl E_col_sc_global;
    Out_gl            dC_out;

    // Developer-only buffers. The production path keeps them null.
    uint8_t*          debug_p3_b_fp4_ptr;
    uint8_t*          debug_p3_b_sc_ptr;
    uint8_t*          debug_gt_fp4_ptr;
    uint8_t*          debug_gt_sc_ptr;
    bf16*             debug_p3_out_ptr;
    float*            debug_p3_out_raw_ptr;
    int               debug_p3_b_fp4_stride;
    int               debug_gt_fp4_stride;
    int               debug_p3_out_stride;
    int               debug_p3_out_raw_stride;
    int               debug_trace_mode;
    int               debug_breakpoint;
    int               debug_block_start;
    int               debug_block_stride;
    int32_t*          debug_trace_ptr;
    int32_t*          debug_trace_count_ptr;
    int               debug_trace_capacity;

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
        Out_sm_tile D[1];
    };
    struct fp4_staging_t {
        Gt_fp4_row_tile Gt_row;
        Gt_sc_row_tile  Gt_row_sc;
    };
    struct p3_tiles_t {
        P3_B_fp4x2_tile B;
    };
    struct p3_scales_t {
        P3_B_sc_tile B_sc;
    };

    __host__ inline dim3 grid() const {
        const int num_vocab_blocks = B.rows() / C::Nb;
        const int num_vocab_superblocks = (num_vocab_blocks + 1) / 2;
        const int num_k_blocks = dC_out.cols() / C::Nb_out;
        const int num_k_superblocks = (num_k_blocks + 1) / 2;
        const bool debug_full_grid_trace = debug_trace_mode >= 2000;
        int total = num_vocab_superblocks * num_k_superblocks;
        const bool debug_single_cluster_trace = debug_trace_mode > 0 && !debug_full_grid_trace;
        int grid_size = debug_single_cluster_trace ? C::CLUSTER_SIZE : min(total, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int phase1_smem = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(output_tiles_t);
        constexpr int fp4_smem = sizeof(fp4_staging_t) + 1024;
        constexpr int p3_smem = sizeof(p3_tiles_t) * 2 + 1024 +
                                sizeof(p3_scales_t) * 2 + 1024;
        constexpr int base_total = phase1_smem + fp4_smem + p3_smem;
        static_assert(base_total + (int)sizeof(Debug_raw_tile) <= MAX_SHARED_MEMORY - 1024);
        const int debug_smem = (debug_p3_out_raw_ptr != nullptr) ? (int)sizeof(Debug_raw_tile) : 0;
        return base_total + debug_smem;
    }
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;

    using P3_B_fp4x2_tile = st_fp4e2m1_2<C::Nb_out/2, C::Mb/2>;
    using P3_B_sc_tile    = st_hf<4, 256, false>;

    using Gt_fp4_row_tile = st_fp4e2m1_2<C::Nb, C::Mb/2>;
    using Gt_sc_row_tile  = st_hf<4, 256, false>;

    using Out_tile       = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH>;
    using Out_sm_tile    = st_bf<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;
    using Debug_raw_tile = st_fl<C::Nb, C::Nb_out / C::EPI_PIPE_DEPTH, false>;

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

    P3_B_fp4x2_gl     E_col;
    P3_B_sc_gl        E_col_sc;
    P3_B_sc_global_gl E_col_sc_global;
    Out_gl            dC_out;

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
        Out_sm_tile D[1];
    };
    struct fp4_staging_t {
        Gt_fp4_row_tile Gt_row;
        Gt_sc_row_tile  Gt_row_sc;
    };
    struct p3_tiles_t {
        P3_B_fp4x2_tile B;
    };
    struct p3_scales_t {
        P3_B_sc_tile B_sc;
    };

    __host__ inline dim3 grid() const {
        const int num_vocab_blocks = B.rows() / C::Nb;
        const int num_vocab_superblocks = (num_vocab_blocks + 1) / 2;
        const int num_k_blocks = dC_out.cols() / C::Nb_out;
        const int num_k_superblocks = (num_k_blocks + 1) / 2;
        int total = num_vocab_superblocks * num_k_superblocks;
        int grid_size = min(total, min(num_sms(), 8));
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int phase1_smem = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                    sizeof(output_tiles_t);
        constexpr int fp4_smem = sizeof(fp4_staging_t) + 1024;
        constexpr int p3_smem = sizeof(p3_tiles_t) * 2 + 1024 +
                                sizeof(p3_scales_t) * 2 + 1024;
        constexpr int base_total = phase1_smem + fp4_smem + p3_smem;
        static_assert(base_total <= MAX_SHARED_MEMORY - 1024);
        return base_total;
    }
};

template <typename C, typename G, bool Debug>
__device__ inline void kernel_impl(const G& g) {

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.E_col.template prefetch_tma<typename G::P3_B_fp4x2_tile>();
        g.E_col_sc.template prefetch_tma<typename G::P3_B_sc_tile>();
        g.dC_out.template prefetch_tma<typename G::Out_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;

    const int num_vocab_blocks = g.B.rows() / C::Nb;
    const int num_vocab_superblocks = (num_vocab_blocks + 1) / 2;
    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_k_blocks = g.dC_out.cols() / C::Nb_out;
    const int num_k_superblocks = (num_k_blocks + 1) / 2;
    const int num_blocks = num_vocab_superblocks * num_k_superblocks;
    const int num_iters_per_row = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_k_superblocks;

    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::output_tiles_t &output_tiles = sm_allocator.allocate<G::output_tiles_t>();
    typename G::fp4_staging_t  &fp4_staging = sm_allocator.allocate<G::fp4_staging_t>();
    typename G::p3_tiles_t     (&p3_tiles)[2] = sm_allocator.allocate<G::p3_tiles_t, 2>();
    typename G::p3_scales_t    (&p3_scales)[2] = sm_allocator.allocate<G::p3_scales_t, 2>();
    typename G::Debug_raw_tile *debug_raw = nullptr;
    if constexpr (Debug) {
        if (g.debug_p3_out_raw_ptr) {
            debug_raw = &sm_allocator.allocate<typename G::Debug_raw_tile>();
        }
    }
    const bool debug_trace_enabled = [&]() {
        if constexpr (Debug) return g.debug_trace_mode > 0;
        else                  return false;
    }();
    const bool debug_trace_full_grid = [&]() {
        if constexpr (Debug) return g.debug_trace_mode >= 2000;
        else                  return false;
    }();
    const bool debug_trace_full_run = [&]() {
        if constexpr (Debug) {
            return (g.debug_trace_mode >= 1000 && g.debug_trace_mode < 2000) ||
                   g.debug_trace_mode >= 3000;
        }
        else                  return false;
    }();
    const int block_loop_start = [&]() {
        if constexpr (Debug) return g.debug_block_stride > 0 ? g.debug_block_start : cluster_id;
        else                  return cluster_id;
    }();
    const int block_loop_stride = [&]() {
        if constexpr (Debug) return g.debug_block_stride > 0 ? g.debug_block_stride : (gridDim.x / C::CLUSTER_SIZE);
        else                  return gridDim.x / C::CLUSTER_SIZE;
    }();
    const int debug_target_block = [&]() {
        if constexpr (Debug) {
            if (g.debug_trace_mode < 0)     return -g.debug_trace_mode - 1;
            if (g.debug_trace_mode >= 3000) return g.debug_trace_mode - 3001;
            if (g.debug_trace_mode >= 2000) return g.debug_trace_mode - 2001;
            if (g.debug_trace_mode >= 1000) return g.debug_trace_mode - 1001;
            if (g.debug_trace_mode > 0)     return g.debug_trace_mode - 1;
            return 0;
        } else {
            return 0;
        }
    }();
    auto should_stop_block = [&](int block_idx) {
        if constexpr (Debug) return debug_trace_enabled && !debug_trace_full_run && block_idx > debug_target_block;
        else                  return false;
    };
    auto should_stop_row_block = [&](int row_block_idx) {
        if constexpr (Debug) return debug_trace_enabled && !debug_trace_full_run && row_block_idx > 1;
        else                  return false;
    };
    auto debug_trap = [&](int marker, int target_row_block_idx, int block_idx, int row_block_idx) {
        if constexpr (Debug) {
            if (g.debug_breakpoint == marker &&
                block_idx == debug_target_block &&
                row_block_idx == target_row_block_idx &&
                warpgroup::warpid() == 0 &&
                (threadIdx.x % 32) == 0) {
                printf("DBG marker=%d cluster=%d cta=%d wg=%d warp=%d block=%d row=%d target=%d\n",
                       marker, cluster_id, cta_id, warpgroup_id, warpgroup::warpid(),
                       block_idx, row_block_idx, debug_target_block);
                asm volatile("trap;");
            }
        }
    };
    auto debug_trap_block = [&](int marker, int block_idx) {
        if constexpr (Debug) {
            if (g.debug_breakpoint == marker &&
                block_idx == debug_target_block &&
                warpgroup::warpid() == 0 &&
                (threadIdx.x % 32) == 0) {
                printf("DBG marker=%d cluster=%d cta=%d wg=%d warp=%d block=%d\n",
                       marker, cluster_id, cta_id, warpgroup_id, warpgroup::warpid(), block_idx);
                asm volatile("trap;");
            }
        }
    };
    auto debug_trap_prev_block = [&](int marker, int block_idx) {
        if constexpr (Debug) {
            if (g.debug_breakpoint == marker &&
                block_idx + 1 == debug_target_block &&
                warpgroup::warpid() == 0 &&
                (threadIdx.x % 32) == 0) {
                printf("DBG marker=%d cluster=%d cta=%d wg=%d warp=%d block=%d prev_target=%d\n",
                       marker, cluster_id, cta_id, warpgroup_id, warpgroup::warpid(),
                       block_idx, debug_target_block);
                asm volatile("trap;");
            }
        }
    };
    auto debug_log_trace = [&](int role, int cta, int warp_in_wg, int marker,
                               int block_idx, int row_block_idx, int subpass,
                               int state0, int state1) {
        if constexpr (Debug) {
            if (g.debug_trace_ptr &&
                g.debug_trace_count_ptr) {
                int *trace_count_ptr = reinterpret_cast<int *>(g.debug_trace_count_ptr);
                const int slot = atomicAdd(trace_count_ptr, 1);
                if (slot < g.debug_trace_capacity) {
                    int32_t *record = g.debug_trace_ptr + slot * DEBUG_TRACE_WIDTH;
                    record[0] = role;
                    record[1] = cta;
                    record[2] = warp_in_wg;
                    record[3] = marker;
                    record[4] = block_idx;
                    record[5] = row_block_idx;
                    record[6] = subpass;
                    record[7] = state0;
                    record[8] = state1;
                }
            }
        }
    };
    auto debug_log_math = [&](int marker, int block_idx, int row_block_idx, int subpass,
                              uint32_t issue_bits, uint32_t acc_bits) {
        if constexpr (Debug) {
            if (cta_id == 0 &&
                warpgroup_id == C::MATH_WARPGROUP_ID &&
                warpgroup::warpid() == 0 &&
                warp::elect_leader()) {
                debug_log_trace(
                    TRACE_ROLE_MATH, cta_id, warpgroup::warpid(), marker,
                    block_idx, row_block_idx, subpass,
                    static_cast<int>(issue_bits), static_cast<int>(acc_bits));
            }
        }
    };
    auto debug_log_epi = [&](int marker, int block_idx, int row_block_idx, int subpass,
                             int state0, int state1) {
        if constexpr (Debug) {
            if (warpgroup_id == C::EPILOGUE_WARPGROUP_ID && (threadIdx.x % 32) == 0) {
                debug_log_trace(
                    TRACE_ROLE_EPILOGUE, cta_id, warpgroup::warpid(), marker,
                    block_idx, row_block_idx, subpass, state0, state1);
            }
        }
    };

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived[2];
    __shared__ semaphore outputs_finished[2];
    __shared__ semaphore a_super_ready[2];
    __shared__ semaphore a_super_consumed;
    __shared__ semaphore a_clear_ready;
    __shared__ semaphore epi_finished;
    __shared__ semaphore epi_block_done;
    __shared__ semaphore epi_kernel_done;
    __shared__ semaphore epi_scratch_released;
    __shared__ semaphore p3_tiles_arrived[2];
    __shared__ semaphore p3_scales_arrived[2];
    __shared__ semaphore p3_inputs_finished[2];
    __shared__ semaphore p3_outputs_arrived[2];
    __shared__ semaphore p3_outputs_finished;

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            init_semaphore(outputs_arrived[s], 0, 1);
            init_semaphore(outputs_finished[s], 0, C::CLUSTER_SIZE);
            init_semaphore(a_super_ready[s], C::CLUSTER_SIZE, 0);
        }
        init_semaphore(a_super_consumed, C::CLUSTER_SIZE, 0);
        init_semaphore(a_clear_ready, C::CLUSTER_SIZE, 0);
        init_semaphore(epi_finished, 0, C::CLUSTER_SIZE);
        init_semaphore(epi_block_done, 0, C::CLUSTER_SIZE);
        init_semaphore(epi_kernel_done, 0, C::CLUSTER_SIZE);
        init_semaphore(epi_scratch_released, 1, 0);
        #pragma unroll
        for (int ksub = 0; ksub < 2; ++ksub) {
            init_semaphore(p3_tiles_arrived[ksub], 0, 1);
            init_semaphore(p3_scales_arrived[ksub], 0, 1);
            init_semaphore(p3_inputs_finished[ksub], 0, 1);
            init_semaphore(p3_outputs_arrived[ksub], 0, 1);
        }
        init_semaphore(p3_outputs_finished, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();
    if constexpr (Debug) {
        if (g.debug_trace_mode > 0 && cta_id == 0 && threadIdx.x == 0) {
            printf("DBG sem_addr tmem=%x inputs0=%x inputs1=%x out_arr0=%x out_fin0=%x p3_in0=%x p3_out0=%x p3_out_fin=%x\n",
                   (unsigned)__cvta_generic_to_shared(&tmem_provisioned),
                   (unsigned)__cvta_generic_to_shared(&inputs_finished[0]),
                   (unsigned)__cvta_generic_to_shared(&inputs_finished[1]),
                   (unsigned)__cvta_generic_to_shared(&outputs_arrived[0]),
                   (unsigned)__cvta_generic_to_shared(&outputs_finished[0]),
                   (unsigned)__cvta_generic_to_shared(&p3_inputs_finished[0]),
                   (unsigned)__cvta_generic_to_shared(&p3_outputs_arrived[0]),
                   (unsigned)__cvta_generic_to_shared(&p3_outputs_finished));
        }
    }

    if (warpgroup_id == C::PRODUCER_WARPGROUP_ID && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            int producer_epi_phase = 0;
            int loaded_blocks = 0;
            int producer_last_block_idx = -1;
            for (int block_idx = block_loop_start; block_idx < num_blocks; block_idx += block_loop_stride) {
                if (should_stop_block(block_idx)) break;
                producer_last_block_idx = block_idx;
                if (loaded_blocks > 0) {
                    wait(epi_finished, producer_epi_phase);
                    producer_epi_phase ^= 1;
                }
                if constexpr (Debug) {
                    if (g.debug_breakpoint == 70 &&
                        block_idx == debug_target_block &&
                        warp_id == 3 &&
                        (threadIdx.x % 32) == 0) {
                        asm volatile("trap;");
                    }
                }
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
                const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
                const int k_superblock_idx = idx_within_supergroup / vocab_superblocks_in_group;
                const int k_block_base = 2 * k_superblock_idx;
                for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                    if (should_stop_row_block(row_block_idx)) break;
                    for (int subpass = 0; subpass < 2; ++subpass) {
                        const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                        if (vocab_block_idx >= num_vocab_blocks) continue;
                        for (int i = 0; i < num_iters_per_row; ++i) {
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
                    }

                    #pragma unroll
                    for (int ksub = 0; ksub < 2; ++ksub) {
                        const int k_block_idx = k_block_base + ksub;
                        if (k_block_idx >= num_k_blocks) continue;
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 73 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 1 &&
                                warp_id == 3 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=73 cluster=%d cta=%d warp=%d block=%d row=%d ksub=%d phase=%u\n",
                                       cluster_id, cta_id, warp_id, block_idx, row_block_idx, ksub,
                                       get_phasebit<1>(phasebits, 6 + ksub));
                                asm volatile("trap;");
                            }
                        }
                        wait(p3_inputs_finished[ksub], get_phasebit<1>(phasebits, 6 + ksub));
                        tma::cluster::load_async(
                            p3_tiles[ksub].B, g.E_col,
                            {k_block_idx * 2 + cta_id, row_block_idx},
                            p3_tiles_arrived[ksub], (uint16_t)(1 << cta_id), 0);
                        update_phasebit<1>(phasebits, 6 + ksub);
                    }
                }
                ++loaded_blocks;
            }
            everyone::tma::cluster::wait();
            if constexpr (Debug) {
                if (g.debug_breakpoint == 130 &&
                    producer_last_block_idx == debug_target_block &&
                    warp_id == 3 &&
                    (threadIdx.x % 32) == 0) {
                    printf("DBG marker=130 cluster=%d cta=%d warp=%d last_block=%d loaded_blocks=%d\n",
                           cluster_id, cta_id, warp_id, producer_last_block_idx, loaded_blocks);
                    asm volatile("trap;");
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();
            everyone::tma::cluster::wait();
            int producer_epi_phase = 0;
            int loaded_blocks = 0;
            int producer_last_block_idx = -1;
            for (int block_idx = block_loop_start; block_idx < num_blocks; block_idx += block_loop_stride) {
                if (should_stop_block(block_idx)) break;
                producer_last_block_idx = block_idx;
                if (loaded_blocks > 0) {
                    wait(epi_finished, producer_epi_phase);
                    producer_epi_phase ^= 1;
                }
                if constexpr (Debug) {
                    if (g.debug_breakpoint == 72 &&
                        block_idx == debug_target_block &&
                        warp_id == 2 &&
                        (threadIdx.x % 32) == 0) {
                        asm volatile("trap;");
                    }
                }
                const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
                const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
                const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
                const int k_superblock_idx = idx_within_supergroup / vocab_superblocks_in_group;
                const int k_block_base = 2 * k_superblock_idx;
                for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                    if (should_stop_row_block(row_block_idx)) break;
                    for (int subpass = 0; subpass < 2; ++subpass) {
                        const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                        if (vocab_block_idx >= num_vocab_blocks) continue;
                        for (int i = 0; i < num_iters_per_row; ++i) {
                            if constexpr (Debug) {
                                if (g.debug_trace_mode > 0 &&
                                    block_idx == debug_target_block &&
                                    row_block_idx == 0 &&
                                    subpass <= 1 &&
                                    i < 2) {
                                    printf("DBG prod_pre_phase1_scales cta=%d warp=%d row=%d subpass=%d iter=%d stage=%u phase=%u\n",
                                           cta_id, warp_id, row_block_idx, subpass, i,
                                           stage, get_phasebit<1>(phasebits, stage));
                                }
                            }
                            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                            tma::cluster::load_async(
                                input_scales[stage].A, g.A_sc,
                                {row_block_idx * 2 + cta_id, i, 0},
                                scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                            tma::cluster::load_async(
                                input_scales[stage].B[0], g.B_sc,
                                {vocab_block_idx, i, 0},
                                scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                            if constexpr (Debug) {
                                if (g.debug_trace_mode > 0 &&
                                    block_idx == debug_target_block &&
                                    row_block_idx == 0 &&
                                    subpass <= 1 &&
                                    i < 2) {
                                    printf("DBG prod_post_phase1_scales cta=%d warp=%d row=%d subpass=%d iter=%d stage=%u next_phase=%u\n",
                                           cta_id, warp_id, row_block_idx, subpass, i,
                                           stage, get_phasebit<1>(phasebits, stage) ^ 1);
                                }
                            }
                            update_phasebit<1>(phasebits, stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                    }

                    #pragma unroll
                    for (int ksub = 0; ksub < 2; ++ksub) {
                        const int k_block_idx = k_block_base + ksub;
                        if (k_block_idx >= num_k_blocks) continue;
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 74 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 1 &&
                                warp_id == 2 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=74 cluster=%d cta=%d warp=%d block=%d row=%d ksub=%d phase=%u\n",
                                       cluster_id, cta_id, warp_id, block_idx, row_block_idx, ksub,
                                       get_phasebit<1>(phasebits, 6 + ksub));
                                asm volatile("trap;");
                            }
                        }
                        wait(p3_inputs_finished[ksub], get_phasebit<1>(phasebits, 6 + ksub));
                        tma::cluster::load_async(
                            p3_scales[ksub].B_sc, g.E_col_sc,
                            {k_block_idx, row_block_idx, 0},
                            p3_scales_arrived[ksub], (uint16_t)(1 << cta_id), 0);
                        update_phasebit<1>(phasebits, 6 + ksub);
                    }
                }
                ++loaded_blocks;
            }
            everyone::tma::cluster::wait();
            if constexpr (Debug) {
                if (g.debug_breakpoint == 131 &&
                    producer_last_block_idx == debug_target_block &&
                    warp_id == 2 &&
                    (threadIdx.x % 32) == 0) {
                    printf("DBG marker=131 cluster=%d cta=%d warp=%d last_block=%d loaded_blocks=%d\n",
                           cluster_id, cta_id, warp_id, producer_last_block_idx, loaded_blocks);
                    asm volatile("trap;");
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
        int a_phase[2] = {0, 0};

        auto phase1_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto phase1_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
        auto p3_tm       = tm_allocator.template allocate<full_tt_fl<C::Nb>>(256);
        constexpr int phase1_sc_offset = 384;
        auto A_sc_tm     = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(phase1_sc_offset);
        auto B_sc_tm     = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
            phase1_sc_offset + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

        constexpr int p3_sc_offset = 384;
        auto p3_A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::P3_A_SCALE_CHUNKS>>(p3_sc_offset);
        auto p3_B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::P3_SCALE_CHUNKS>>(
            p3_sc_offset + 4 * C::P3_A_SCALE_CHUNKS);

	        const int lane_id = threadIdx.x % 32;
	        const int wg_thread = warpgroup::warpid() * WARP_THREADS + lane_id;
	        using out_rt    = rt_fl<C::Nb / 4, C::Nb_out / C::EPI_PIPE_DEPTH>;
	        using out_rt_bf = rt_bf<C::Nb / 4, C::Nb_out / C::EPI_PIPE_DEPTH>;
        constexpr float kFp4Max = 6.0f;
        constexpr float kE4M3Max = 448.0f;
        const float gt_row_sg_val = fmaxf(g.grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
        const float p3_scale = gt_row_sg_val * g.E_col_sc_global[{0}];

        uint32_t issue_stage = 0;
        uint32_t issue_phasebits = 0xFFFF0000;
        uint32_t acc_phasebits = 0xFFFF0000;
        bool p3_tm_live = false;
        int produced_blocks = 0;
        int last_block_idx = -1;
        int epi_phase = 0;
        int epi_done_phase = 0;
        auto do_phase1 = [&](auto &accum, int debug_block_idx, int debug_row_block_idx, int debug_subpass) {
            for (int i = 0; i < num_iters_per_row; ++i) {
                if (warpgroup::warpid() == 0) {
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_pre_phase1_scales cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u phase=%u\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, get_phasebit<0>(issue_phasebits, issue_stage));
                        }
                    }
                    if (lane_id == 0) {
                        tma::expect_bytes(scales_arrived[issue_stage], 2 * sizeof(typename G::input_scales_t));
                    }
                    __syncwarp();
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_post_phase1_scales_expect cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u phase=%u\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, get_phasebit<0>(issue_phasebits, issue_stage));
                        }
                    }
                    wait(scales_arrived[issue_stage], get_phasebit<0>(issue_phasebits, issue_stage));
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_post_phase1_scales cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u phase=%u\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, get_phasebit<0>(issue_phasebits, issue_stage));
                        }
                    }
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

                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_pre_phase1_tiles cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u phase=%u\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, get_phasebit<0>(issue_phasebits, issue_stage));
                        }
                    }
                    if (lane_id == 0) {
                        tma::expect_bytes(tiles_arrived[issue_stage], 2 * sizeof(typename G::input_tiles_t));
                    }
                    __syncwarp();
                    wait(tiles_arrived[issue_stage], get_phasebit<0>(issue_phasebits, issue_stage));
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_post_phase1_tiles cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u phase=%u\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, get_phasebit<0>(issue_phasebits, issue_stage));
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_pre_phase1_mma cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u first=%d\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, i == 0 ? 1 : 0);
                        }
                    }
                    if (warp::elect_leader()) {
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
                    }
                    __syncwarp();
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            debug_block_idx == debug_target_block &&
                            debug_row_block_idx <= 1 &&
                            debug_subpass <= 1 &&
                            i < 2 &&
                            lane_id == 0) {
                            printf("DBG math_post_phase1_mma cta=%d warp=%d block=%d row=%d subpass=%d iter=%d stage=%u first=%d\n",
                                   cta_id, warpgroup::warpid(), debug_block_idx, debug_row_block_idx,
                                   debug_subpass, i, issue_stage, i == 0 ? 1 : 0);
                        }
                    }
                    update_phasebit<0>(issue_phasebits, issue_stage);
                    issue_stage = (issue_stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        };

        for (int block_idx = block_loop_start; block_idx < num_blocks; block_idx += block_loop_stride) {
            if (should_stop_block(block_idx)) break;
            last_block_idx = block_idx;
            if (produced_blocks > 0) {
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        cta_id == 1 &&
                        lane_id == 0) {
                        printf("DBG math_pre_epi_finished cta=%d warp=%d block=%d epi_phase=%d issue=%08x acc=%08x p3_live=%d\n",
                               cta_id, warpgroup::warpid(), block_idx, epi_phase,
                               issue_phasebits, acc_phasebits, p3_tm_live ? 1 : 0);
                    }
                }
                wait(epi_finished, epi_phase);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        cta_id == 1 &&
                        lane_id == 0) {
                        printf("DBG math_post_epi_finished cta=%d warp=%d block=%d epi_phase=%d issue=%08x acc=%08x p3_live=%d\n",
                               cta_id, warpgroup::warpid(), block_idx, epi_phase,
                               issue_phasebits, acc_phasebits, p3_tm_live ? 1 : 0);
                    }
                }
                epi_phase ^= 1;
                // The block epilogue drains the phase-3 TMEM slot before signaling epi_finished.
                // Consume the pending phase-3 recycle completion here so the next block starts
                // with both the barrier state and the local issue phasebit aligned.
                if (p3_tm_live) {
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block &&
                            cta_id == 1 &&
                            lane_id == 0) {
                            printf("DBG math_pre_block_p3_recycle_wait cta=%d warp=%d block=%d issue=%08x acc=%08x phase=%u\n",
                                   cta_id, warpgroup::warpid(), block_idx,
                                   issue_phasebits, acc_phasebits, get_phasebit<0>(issue_phasebits, 8));
                        }
                    }
                    wait(p3_outputs_finished, get_phasebit<0>(issue_phasebits, 8));
                    tensor_after_thread_sync();
                    warpgroup::sync(1);
                    update_phasebit<0>(issue_phasebits, 8);
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block &&
                            cta_id == 1 &&
                            lane_id == 0) {
                            printf("DBG math_post_block_p3_recycle_wait cta=%d warp=%d block=%d issue=%08x acc=%08x phase=%u\n",
                                   cta_id, warpgroup::warpid(), block_idx,
                                   issue_phasebits, acc_phasebits, get_phasebit<0>(issue_phasebits, 8));
                        }
                    }
                    p3_tm_live = false;
                }
            }
            const int supergroup_idx = block_idx / num_blocks_per_supergroup;
            const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
	            const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
	            const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
            const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
            const int k_superblock_idx = idx_within_supergroup / vocab_superblocks_in_group;
            const int k_block_base = 2 * k_superblock_idx;
            if constexpr (Debug) {
                if (g.debug_trace_mode > 0 &&
                    block_idx == debug_target_block &&
                    cta_id == 1 &&
                    lane_id == 0) {
                    printf("DBG math_block_entry_cta1 cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                           cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                }
            }
            debug_log_math(TRACE_BLOCK_ENTRY, block_idx, -1, -1, issue_phasebits, acc_phasebits);
	            if constexpr (Debug) {
	                if (g.debug_breakpoint == 109 &&
	                    block_idx == debug_target_block &&
	                    cta_id == 0 &&
	                    warpgroup::warpid() == 0 &&
	                    warp::elect_leader()) {
	                    printf("DBG marker=109 cluster=%d cta=%d block=%d issue=%08x acc=%08x\n",
	                           cluster_id, cta_id, block_idx, issue_phasebits, acc_phasebits);
	                    asm volatile("trap;");
	                }
	            }
	            out_rt D_acc[2][C::EPI_PIPE_DEPTH];

		            for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
		                if (should_stop_row_block(row_block_idx)) break;
		                bool valid_subpass[2] = {false, false};
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                cta_id == 1 &&
                                lane_id == 0) {
                                printf("DBG math_row_entry_cta1 cta=%d warp=%d block=%d row=%d issue=%08x acc=%08x\n",
                                       cta_id, warpgroup::warpid(), block_idx, row_block_idx,
                                       issue_phasebits, acc_phasebits);
                            }
                        }
                        debug_log_math(TRACE_ROW_ENTRY, block_idx, row_block_idx, -1, issue_phasebits, acc_phasebits);
		                if constexpr (Debug) {
		                    if (g.debug_breakpoint == 110 &&
		                        block_idx == debug_target_block &&
		                        row_block_idx == 0 &&
		                        cta_id == 0 &&
		                        warpgroup::warpid() == 0 &&
		                        warp::elect_leader()) {
		                        printf("DBG marker=110 cluster=%d cta=%d block=%d row=%d issue=%08x acc=%08x\n",
		                               cluster_id, cta_id, block_idx, row_block_idx, issue_phasebits, acc_phasebits);
		                        asm volatile("trap;");
		                    }
		                }
	                debug_trap(30, 0, block_idx, row_block_idx);
	                debug_trap(20, 1, block_idx, row_block_idx);

	                if constexpr (Debug) {
	                    if (g.debug_breakpoint == 111 &&
	                        block_idx == debug_target_block &&
	                        row_block_idx == 0 &&
	                        cta_id == 0 &&
	                        warpgroup::warpid() == 0 &&
	                        warp::elect_leader()) {
	                        printf("DBG marker=111 cluster=%d cta=%d block=%d row=%d issue=%08x acc=%08x\n",
	                               cluster_id, cta_id, block_idx, row_block_idx, issue_phasebits, acc_phasebits);
	                        asm volatile("trap;");
	                    }
	                }
                    for (int subpass = 0; subpass < 2; ++subpass) {
                        const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                        valid_subpass[subpass] = vocab_block_idx < num_vocab_blocks;
                        if (!valid_subpass[subpass]) continue;
                        if (cta_id == 0) {
                            warpgroup::sync(1);
                            if constexpr (Debug) {
                                if (warp::elect_leader() &&
                                    warpgroup::warpid() == 0 &&
                                    g.debug_trace_mode > 0 &&
                                    block_idx == debug_target_block &&
                                    row_block_idx <= 1 &&
                                    subpass <= 1) {
                                    printf("DBG math_pre_outputs_finished cta=%d warp=%d row=%d subpass=%d issue=%08x out_phase=%u\n",
                                           cta_id, warpgroup::warpid(), row_block_idx, subpass,
                                           issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                }
                                if (warp::elect_leader() &&
                                    warpgroup::warpid() == 0 &&
                                    g.debug_breakpoint == 87 &&
                                    block_idx + 1 == debug_target_block) {
                                    printf("DBG marker=87 cluster=%d cta=%d block=%d row=%d subpass=%d issue=%08x out_p=%u\n",
                                           cluster_id, cta_id, block_idx, row_block_idx, subpass,
                                           issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                    asm volatile("trap;");
                                }
                                if (warp::elect_leader() &&
                                    warpgroup::warpid() == 0 &&
                                    g.debug_breakpoint == 107 &&
                                    block_idx == debug_target_block) {
                                    printf("DBG marker=107 cluster=%d cta=%d block=%d row=%d subpass=%d issue=%08x out_p=%u\n",
                                           cluster_id, cta_id, block_idx, row_block_idx, subpass,
                                           issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                    asm volatile("trap;");
                                }
                            }
                            debug_log_math(TRACE_PRE_OUTPUTS_FINISHED, block_idx, row_block_idx, subpass, issue_phasebits, acc_phasebits);
                            wait(outputs_finished[subpass], get_phasebit<1>(issue_phasebits, 9 + subpass));
                            tensor_after_thread_sync();
                            debug_log_math(TRACE_POST_OUTPUTS_FINISHED, block_idx, row_block_idx, subpass, issue_phasebits, acc_phasebits);
                            if constexpr (Debug) {
                                if (warp::elect_leader() &&
                                    warpgroup::warpid() == 0 &&
                                    g.debug_trace_mode > 0 &&
                                    block_idx == debug_target_block &&
                                    row_block_idx <= 1 &&
                                    subpass <= 1) {
                                    printf("DBG math_post_outputs_finished cta=%d warp=%d row=%d subpass=%d issue=%08x out_phase=%u\n",
                                           cta_id, warpgroup::warpid(), row_block_idx, subpass,
                                           issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                }
                                if (warp::elect_leader() &&
                                    warpgroup::warpid() == 0 &&
                                    g.debug_breakpoint == 88 &&
                                    block_idx + 1 == debug_target_block) {
                                    printf("DBG marker=88 cluster=%d cta=%d block=%d row=%d subpass=%d issue=%08x out_p=%u\n",
                                           cluster_id, cta_id, block_idx, row_block_idx, subpass,
                                           issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                    asm volatile("trap;");
                                }
                                if (warp::elect_leader() &&
                                    warpgroup::warpid() == 0 &&
                                    g.debug_breakpoint == 108 &&
                                    block_idx == debug_target_block) {
                                    printf("DBG marker=108 cluster=%d cta=%d block=%d row=%d subpass=%d issue=%08x out_p=%u\n",
                                           cluster_id, cta_id, block_idx, row_block_idx, subpass,
                                           issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                    asm volatile("trap;");
                                }
                            }
                            auto &phase1_tm = (subpass == 0) ? phase1_tm_0 : phase1_tm_1;
                            do_phase1(phase1_tm, block_idx, row_block_idx, subpass);
                            warpgroup::sync(1);
                            if (warp::elect_leader() && warpgroup::warpid() == 0) {
                                if constexpr (Debug) {
                                    if (g.debug_trace_mode > 0 &&
                                        block_idx == debug_target_block &&
                                        row_block_idx <= 1 &&
                                        subpass <= 1) {
                                        printf("DBG math_pre_outputs_arrived_commit cta=%d warp=%d row=%d subpass=%d issue=%08x out_phase=%u\n",
                                               cta_id, warpgroup::warpid(), row_block_idx, subpass,
                                               issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                    }
                                }
                                tensor_commit<2>(outputs_arrived[subpass]);
                                tensor_after_thread_sync();
                                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                                if constexpr (Debug) {
                                    if (g.debug_trace_mode > 0 &&
                                        block_idx == debug_target_block &&
                                        row_block_idx <= 1 &&
                                        subpass <= 1) {
                                        printf("DBG math_post_outputs_arrived_commit cta=%d warp=%d row=%d subpass=%d issue=%08x next_out_phase=%u\n",
                                               cta_id, warpgroup::warpid(), row_block_idx, subpass,
                                               issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                    }
                                    if (g.debug_breakpoint == 89 &&
                                        block_idx + 1 == debug_target_block) {
                                        printf("DBG marker=89 cluster=%d cta=%d block=%d row=%d subpass=%d issue=%08x next_out_p=%u\n",
                                               cluster_id, cta_id, block_idx, row_block_idx, subpass,
                                               issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                                        asm volatile("trap;");
                                    }
                                }
                            }
                            warpgroup::sync(1);
                            update_phasebit<1>(issue_phasebits, 9 + subpass);
                        }
                    }

                    #pragma unroll
                    for (int subpass = 0; subpass < 2; ++subpass) {
                        if (!valid_subpass[subpass]) continue;
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                subpass <= 1 &&
                                cta_id == 1 &&
                                lane_id == 0) {
                                printf("DBG math_pre_a_super_ready cta=%d warp=%d row=%d subpass=%d a_phase=%d issue=%08x acc=%08x\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, subpass, a_phase[subpass],
                                       issue_phasebits, acc_phasebits);
                            }
                        }
                        wait(a_super_ready[subpass], a_phase[subpass]);
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                subpass <= 1 &&
                                cta_id == 1 &&
                                lane_id == 0) {
                                printf("DBG math_post_a_super_ready cta=%d warp=%d row=%d subpass=%d a_phase=%d issue=%08x acc=%08x\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, subpass, a_phase[subpass],
                                       issue_phasebits, acc_phasebits);
                            }
                        }
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 90 &&
                                block_idx + 1 == debug_target_block) {
                                printf("DBG marker=90 cluster=%d cta=%d block=%d row=%d subpass=%d a_phase=%d\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, subpass, a_phase[subpass]);
                                asm volatile("trap;");
                            }
                        }
                        a_phase[subpass] ^= 1;
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 91 &&
                            block_idx + 1 == debug_target_block) {
                            printf("DBG marker=91 cluster=%d cta=%d block=%d row=%d a_phase0=%d a_phase1=%d\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, a_phase[0], a_phase[1]);
                            asm volatile("trap;");
                        }
                    }
                    debug_trap_prev_block(33, block_idx);
                    debug_trap(31, 0, block_idx, row_block_idx);
                    debug_trap(21, 1, block_idx, row_block_idx);

                    if (cta_id == 0) {
                        #pragma unroll
                        for (int ii = 0; ii < C::P3_A_SCALE_CHUNKS; ++ii) {
                            auto p3_A_sc_sub = p3_A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                            auto &A_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&fp4_staging.Gt_row_sc.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(p3_A_sc_sub, A_sc_sm_sub);
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 92 &&
                            block_idx + 1 == debug_target_block) {
                            printf("DBG marker=92 cluster=%d cta=%d block=%d row=%d issue=%08x a_phase0=%d a_phase1=%d\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, issue_phasebits, a_phase[0], a_phase[1]);
                            asm volatile("trap;");
                        }
                    }

                debug_trap_prev_block(35, block_idx);
                for (int ksub = 0; ksub < 2; ++ksub) {
                    const int k_block_idx = k_block_base + ksub;
                    if (k_block_idx >= num_k_blocks) continue;
                    if (p3_tm_live) {
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub <= 1 &&
                                lane_id == 0) {
                                printf("DBG math_pre_p3_recycle_wait cta=%d warp=%d row=%d ksub=%d issue=%08x phase=%u\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                       issue_phasebits, get_phasebit<0>(issue_phasebits, 8));
                            }
                        }
                        wait(p3_outputs_finished, get_phasebit<0>(issue_phasebits, 8));
                        tensor_after_thread_sync();
                        warpgroup::sync(1);
                        update_phasebit<0>(issue_phasebits, 8);
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub <= 1 &&
                                lane_id == 0) {
                                printf("DBG math_post_p3_recycle_wait cta=%d warp=%d row=%d ksub=%d issue=%08x phase=%u\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                       issue_phasebits, get_phasebit<0>(issue_phasebits, 8));
                            }
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 83 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 1 &&
                            cta_id == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=83 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x acc=%08x\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub, issue_phasebits, acc_phasebits);
                            asm volatile("trap;");
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 36 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=36 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x acc=%08x issue_p=%u acc_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   issue_phasebits, acc_phasebits,
                                   get_phasebit<0>(issue_phasebits, 6 + ksub),
                                   get_phasebit<0>(acc_phasebits, 11 + ksub));
                        }
                        if (g.debug_breakpoint == 37 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 1 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=37 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x acc=%08x issue_p=%u acc_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   issue_phasebits, acc_phasebits,
                                   get_phasebit<0>(issue_phasebits, 6 + ksub),
                                   get_phasebit<0>(acc_phasebits, 11 + ksub));
                        }
                    }
                    if (ksub == 0) debug_trap_prev_block(36, block_idx);
                    else           debug_trap_prev_block(37, block_idx);

                    if constexpr (Debug) {
                        if (warp::elect_leader() &&
                            warpgroup::warpid() == 0 &&
                            g.debug_breakpoint == 79 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 1) {
                            printf("DBG marker=79 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x wait_sc_p=%u wait_tile_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   issue_phasebits,
                                   get_phasebit<0>(issue_phasebits, 6 + ksub),
                                   get_phasebit<0>(issue_phasebits, 6 + ksub));
                            asm volatile("trap;");
                        }
                        if (warp::elect_leader() &&
                            warpgroup::warpid() == 0 &&
                            g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block &&
                            row_block_idx == 0 &&
                            ksub <= 1) {
                            printf("DBG math_pre_p3_scales_wait cta=%d warp=%d row=%d ksub=%d issue=%08x phase=%u\n",
                                   cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                   issue_phasebits, get_phasebit<0>(issue_phasebits, 6 + ksub));
                        }
                    }
                    if (cta_id == 0) {
                        if (warpgroup::warpid() == 0 && lane_id == 0) {
                            tma::expect_bytes(p3_scales_arrived[ksub], 2 * sizeof(typename G::p3_scales_t));
                        }
                        wait(p3_scales_arrived[ksub], get_phasebit<0>(issue_phasebits, 6 + ksub));
                        if constexpr (Debug) {
                            if (warp::elect_leader() &&
                                warpgroup::warpid() == 0 &&
                                g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub <= 1) {
                                printf("DBG math_post_p3_scales_wait cta=%d warp=%d row=%d ksub=%d issue=%08x phase=%u\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                       issue_phasebits, get_phasebit<0>(issue_phasebits, 6 + ksub));
                            }
                            if (warp::elect_leader() &&
                                warpgroup::warpid() == 0 &&
                                g.debug_breakpoint == 80 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 1) {
                                printf("DBG marker=80 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, issue_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        #pragma unroll
                        for (int ii = 0; ii < C::P3_SCALE_CHUNKS; ++ii) {
                            auto p3_B_sc_sub = p3_B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                            auto &B_sc_sm_sub = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&p3_scales[ksub].B_sc.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(p3_B_sc_sub, B_sc_sm_sub);
                        }

                        if (warpgroup::warpid() == 0 && lane_id == 0) {
                            tma::expect_bytes(p3_tiles_arrived[ksub], 2 * sizeof(typename G::p3_tiles_t));
                        }
                        wait(p3_tiles_arrived[ksub], get_phasebit<0>(issue_phasebits, 6 + ksub));
                        if constexpr (Debug) {
                            if (warp::elect_leader() &&
                                warpgroup::warpid() == 0 &&
                                g.debug_breakpoint == 81 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 1) {
                                printf("DBG marker=81 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, issue_phasebits);
                                asm volatile("trap;");
                            }
                            if (warp::elect_leader() &&
                                warpgroup::warpid() == 0 &&
                                g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub <= 1) {
                                printf("DBG math_post_p3_tiles_wait cta=%d warp=%d row=%d ksub=%d issue=%08x phase=%u\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                       issue_phasebits, get_phasebit<0>(issue_phasebits, 6 + ksub));
                                }
                            if (warp::elect_leader() &&
                                warpgroup::warpid() == 0 &&
                                g.debug_p3_b_fp4_ptr &&
                                cluster_id == 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub == 0) {
                                auto *fp4_bytes = reinterpret_cast<const uint8_t*>(&p3_tiles[ksub].B.data[0]);
                                for (int idx = 0; idx < (int)sizeof(typename G::P3_B_fp4x2_tile); ++idx) {
                                    const int row = idx / g.debug_p3_b_fp4_stride;
                                    const int col = idx % g.debug_p3_b_fp4_stride;
                                    g.debug_p3_b_fp4_ptr[row * g.debug_p3_b_fp4_stride + col] = fp4_bytes[idx];
                                }
                                auto *sc_bytes = reinterpret_cast<const uint8_t*>(&p3_scales[ksub].B_sc.data[0]);
                                for (int idx = 0; idx < (int)sizeof(typename G::P3_B_sc_tile); ++idx) {
                                    g.debug_p3_b_sc_ptr[idx] = sc_bytes[idx];
                                }
                            }
                        }
                        warpgroup::sync(1);
                        if (warp::elect_leader() && warpgroup::warpid() == 0) {
                            if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub <= 1) {
                                printf("DBG math_pre_p3_mma cta=%d warp=%d row=%d ksub=%d issue=%08x acc=%08x\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                       issue_phasebits, acc_phasebits);
                            }
                            }
                            mm2_ABt(
                                p3_tm, fp4_staging.Gt_row, p3_tiles[ksub].B,
                                p3_A_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_A_SCALE_CHUNKS * 16>>(0),
                                p3_B_sc_tm.template subtile<full_tt_fp8e4m3<C::P3_SCALE_CHUNKS * 32>>(0),
                                p3_inputs_finished[ksub]);
                            if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub <= 1) {
                                printf("DBG math_post_p3_mma cta=%d warp=%d row=%d ksub=%d issue=%08x acc=%08x\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                       issue_phasebits, acc_phasebits);
                                    }
                            }
                            tensor_commit<2>(p3_outputs_arrived[ksub]);
                            tensor_after_thread_sync();
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                            if constexpr (Debug) {
                                if (g.debug_breakpoint == 82 &&
                                    block_idx + 1 == debug_target_block &&
                                    ksub == 1) {
                                    printf("DBG marker=82 cluster=%d cta=%d block=%d row=%d ksub=%d issue=%08x\n",
                                           cluster_id, cta_id, block_idx, row_block_idx, ksub, issue_phasebits);
                                    asm volatile("trap;");
                                }
                            }
                        }
                        warpgroup::sync(1);
                        update_phasebit<0>(issue_phasebits, 6 + ksub);
                    }

                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 93 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            cta_id == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=93 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x wait_out_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                            asm volatile("trap;");
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block &&
                            row_block_idx == 0 &&
                            ksub <= 1 &&
                            lane_id == 0) {
                            printf("DBG math_pre_p3_outputs_wait cta=%d warp=%d row=%d ksub=%d acc=%08x phase=%u\n",
                                   cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                        }
                    }
                    wait(p3_outputs_arrived[ksub], get_phasebit<0>(acc_phasebits, 11 + ksub));
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block &&
                            row_block_idx == 0 &&
                            ksub <= 1 &&
                            lane_id == 0) {
                            printf("DBG math_post_p3_outputs_wait cta=%d warp=%d row=%d ksub=%d acc=%08x phase=%u\n",
                                   cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 94 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            cta_id == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=94 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x wait_out_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                            asm volatile("trap;");
                        }
                    }
                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                        out_rt D_reg_fl;
                        warpgroup::load_async(
                            D_reg_fl,
                            p3_tm.template subtile<full_tt_fl<C::Nb_out / C::EPI_PIPE_DEPTH>>(
                                0,
                                epi * (C::Nb_out / C::EPI_PIPE_DEPTH)));
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 103 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 2 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=103 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 104 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 2 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=104 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 95 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 0 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=95 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        warpgroup::sync(1);
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 96 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 0 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=96 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                            if (g.debug_breakpoint == 105 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 2 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=105 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        if constexpr (Debug) {
                            if (g.debug_p3_out_raw_ptr &&
                                cluster_id == 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx == 0 &&
                                ksub == 0) {
	                                warpgroup::store(*debug_raw, D_reg_fl);
	                                warpgroup::sync(1);
	                                const uint32_t raw_base = static_cast<uint32_t>(__cvta_generic_to_shared(&debug_raw->data[0]));
	                                constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
	                                for (int idx = wg_thread; idx < C::Nb * (C::Nb_out / C::EPI_PIPE_DEPTH); idx += CONSUMER_THREADS) {
	                                    const int row = idx / (C::Nb_out / C::EPI_PIPE_DEPTH);
	                                    const int col = idx % (C::Nb_out / C::EPI_PIPE_DEPTH);
	                                    float value;
                                    move<float>::lds(value, G::Debug_raw_tile::idx(raw_base, {row, col}));
                                    g.debug_p3_out_raw_ptr[(cta_id * C::Nb + row) * g.debug_p3_out_raw_stride +
                                                           epi * (C::Nb_out / C::EPI_PIPE_DEPTH) + col] = value;
                                }
                                warpgroup::sync(1);
                            }
                        }
                        warp::mul(D_reg_fl, D_reg_fl, p3_scale);
                        if (row_block_idx == 0) warp::copy(D_acc[ksub][epi], D_reg_fl);
                        else                    warp::add(D_acc[ksub][epi], D_acc[ksub][epi], D_reg_fl);
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 106 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 2 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=106 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        warpgroup::sync(1);
                        if constexpr (Debug) {
                            if (g.debug_breakpoint == 97 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 0 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=97 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                            if (g.debug_breakpoint == 100 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 1 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=100 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                            if (g.debug_breakpoint == 101 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 2 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=101 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                            if (g.debug_breakpoint == 102 &&
                                block_idx + 1 == debug_target_block &&
                                ksub == 0 &&
                                epi == 3 &&
                                cta_id == 0 &&
                                warpgroup::warpid() == 0 &&
                                (threadIdx.x % 32) == 0) {
                                printf("DBG marker=102 cluster=%d cta=%d block=%d row=%d ksub=%d epi=%d acc=%08x\n",
                                       cluster_id, cta_id, block_idx, row_block_idx, ksub, epi, acc_phasebits);
                                asm volatile("trap;");
                            }
                        }
                        tensor_after_thread_sync();
                    }
                    warpgroup::sync(1);
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    if (warpgroup::warpid() == 0 && lane_id == 0) {
                        arrive(p3_outputs_finished, 1);
                        arrive_remote_cluster_v5(p3_outputs_finished, 1 - cta_id, 1);
                    }
                    warpgroup::sync(1);
                    p3_tm_live = true;
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block &&
                            row_block_idx == 0 &&
                            ksub <= 1 &&
                            lane_id == 0) {
                            printf("DBG math_post_p3_recycle_arrive cta=%d warp=%d row=%d ksub=%d issue=%08x phase=%u\n",
                                   cta_id, warpgroup::warpid(), row_block_idx, ksub,
                                   issue_phasebits, get_phasebit<0>(issue_phasebits, 8));
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 98 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            cta_id == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=98 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub, acc_phasebits);
                            asm volatile("trap;");
                        }
                    }
                    update_phasebit<0>(acc_phasebits, 11 + ksub);
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 99 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            cta_id == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=99 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x next_acc_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                            asm volatile("trap;");
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 86 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            cta_id == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=86 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x next_acc_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                            asm volatile("trap;");
                        }
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 38 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 0 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=38 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x next_acc_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                        }
                        if (g.debug_breakpoint == 39 &&
                            block_idx + 1 == debug_target_block &&
                            ksub == 1 &&
                            warpgroup::warpid() == 0 &&
                            (threadIdx.x % 32) == 0) {
                            printf("DBG marker=39 cluster=%d cta=%d block=%d row=%d ksub=%d acc=%08x next_acc_p=%u\n",
                                   cluster_id, cta_id, block_idx, row_block_idx, ksub,
                                   acc_phasebits, get_phasebit<0>(acc_phasebits, 11 + ksub));
                        }
                    }
                    if (ksub == 0) debug_trap_prev_block(38, block_idx);
                    else           debug_trap_prev_block(39, block_idx);
                }

                // The next epilogue row will clear Gt_row / Gt_row_sc as soon as it
                // observes a_super_consumed. Re-converge the math WG and make the
                // row scratch visibility explicit before handing that token over.
                warpgroup::sync(1);
                if (warpgroup::warpid() == 0 && lane_id == 0) {
                    if (row_block_idx + 1 < num_row_blocks) {
                        __threadfence_block();
                        // This row-complete signal returns ownership of staged G^T
                        // scratch to the local epilogue WG before the next row.
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                        arrive(a_super_consumed, 1);
                        arrive_remote_cluster_v5(a_super_consumed, 1 - cta_id, 1);
                    }
                    if constexpr (Debug) {
                        if (g.debug_breakpoint == 22 &&
                            block_idx == debug_target_block &&
                            row_block_idx == 0) {
                            asm volatile("trap;");
                        }
	                    }
	                }
                warpgroup::sync(1);
	                debug_trap_prev_block(34, block_idx);
                    debug_log_math(TRACE_ROW_COMPLETE, block_idx, row_block_idx, -1, issue_phasebits, acc_phasebits);
	            }

	            if (num_row_blocks > 0) {
	                // The split epilogue still owns output_tiles.D[0] for the last subpass scratch
	                // work on both CTAs. Drain the final outputs_finished handoff locally before
	                // either CTA enters the math-owned final D_acc store path.
	                #pragma unroll
	                for (int subpass = 0; subpass < 2; ++subpass) {
	                    const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
	                    if (vocab_block_idx >= num_vocab_blocks) continue;
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                lane_id == 0) {
                                printf("DBG math_pre_tail_outputs_finished cta=%d warp=%d block=%d subpass=%d issue=%08x phase=%u\n",
                                       cta_id, warpgroup::warpid(), block_idx, subpass,
                                       issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                            }
                        }
	                    wait(outputs_finished[subpass], get_phasebit<1>(issue_phasebits, 9 + subpass));
	                    tensor_after_thread_sync();
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 &&
                                block_idx == debug_target_block &&
                                lane_id == 0) {
                                printf("DBG math_post_tail_outputs_finished cta=%d warp=%d block=%d subpass=%d issue=%08x phase=%u\n",
                                       cta_id, warpgroup::warpid(), block_idx, subpass,
                                       issue_phasebits, get_phasebit<1>(issue_phasebits, 9 + subpass));
                            }
                        }
	                }
	            }
	            warpgroup::sync(1);
                wait(epi_block_done, epi_done_phase);
                epi_done_phase ^= 1;
                warpgroup::sync(1);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        lane_id == 0) {
                        printf("DBG math_post_tail_sync cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                               cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                    }
                    // Skip the block-tail trace record here; it runs on cta0/warp0 only and
                    // has become perturbative while we localize the first final-store handoff.
                }

                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        warpgroup::warpid() == 0 &&
                        lane_id == 0) {
                        printf("DBG math_pre_final_trace cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                               cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                    }
                    debug_log_math(TRACE_FINAL_STORE_ENTRY, block_idx, -1, -1, issue_phasebits, acc_phasebits);
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        warpgroup::warpid() == 0 &&
                        lane_id == 0) {
                        printf("DBG math_post_final_trace cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                               cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                    }
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        lane_id == 0) {
                        printf("DBG math_pre_final_store cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                               cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                    }
                }
	            const int vocab_superblock_base = vocab_superblock_idx * 2;
	            #pragma unroll
	            for (int ksub = 0; ksub < 2; ++ksub) {
                const int k_block_idx = k_block_base + ksub;
                if (k_block_idx >= num_k_blocks) continue;
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    out_rt_bf D_reg_bf;
                    warp::copy(D_reg_bf, D_acc[ksub][epi]);
	                    debug_trap_block(50, block_idx);
	                    debug_trap_prev_block(60, block_idx);
	                    warpgroup::tma::store_async_read_wait<0>();
	                    debug_trap_block(51, block_idx);
	                    debug_trap_prev_block(61, block_idx);
	                    warpgroup::sync(1);
	                    debug_trap_prev_block(63, block_idx);
	                    warpgroup::store(output_tiles.D[0], D_reg_bf);
	                    warpgroup::sync(1);
	                    debug_trap_prev_block(64, block_idx);

                    if constexpr (Debug) {
                        if (g.debug_p3_out_ptr &&
                            cluster_id == 0 &&
                            block_idx == debug_target_block &&
                            ksub == 0) {
	                            const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
	                            constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
	                            for (int idx = wg_thread; idx < C::Nb * (C::Nb_out / C::EPI_PIPE_DEPTH); idx += CONSUMER_THREADS) {
	                                const int row = idx / (C::Nb_out / C::EPI_PIPE_DEPTH);
	                                const int col = idx % (C::Nb_out / C::EPI_PIPE_DEPTH);
	                                bf16 value;
                                move<bf16>::lds(value, G::Out_sm_tile::idx(d_base, {row, col}));
                                g.debug_p3_out_ptr[(cta_id * C::Nb + row) * g.debug_p3_out_stride +
                                                   epi * (C::Nb_out / C::EPI_PIPE_DEPTH) + col] = value;
                            }
                            warpgroup::sync(1);
                        }
                    }

                    warpgroup::sync(1);

                    if (vocab_superblock_base + cta_id < num_vocab_blocks) {
	                        const int global_vocab_base = (vocab_superblock_base + cta_id) * C::Nb;
	                        const int global_k_base = (C::EPI_PIPE_DEPTH * k_block_idx + epi) * (C::Nb_out / C::EPI_PIPE_DEPTH);
	                        constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
	                        const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
	                        for (int idx = wg_thread; idx < C::Nb * (C::Nb_out / C::EPI_PIPE_DEPTH); idx += CONSUMER_THREADS) {
	                            const int row = idx / (C::Nb_out / C::EPI_PIPE_DEPTH);
	                            const int col = idx % (C::Nb_out / C::EPI_PIPE_DEPTH);
	                            bf16 value_f;
                            move<bf16>::lds(value_f, G::Out_sm_tile::idx(d_base, {row, col}));
	                            g.dC_out.raw_ptr[(global_vocab_base + row) * g.dC_out.cols() + (global_k_base + col)] =
	                                value_f;
	                        }
	                        warpgroup::sync(1);
	                        debug_trap_prev_block(65, block_idx);
	                    }
	                }
	                debug_trap_prev_block(66, block_idx);
	            }
                if constexpr (Debug) {
                    debug_log_math(TRACE_FINAL_STORE_EXIT, block_idx, -1, -1, issue_phasebits, acc_phasebits);
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        lane_id == 0) {
                        printf("DBG math_post_final_store cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                               cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                    }
                }
                warpgroup::sync(1);
                warpgroup::tma::store_async_read_wait<0>();
                // In the split pipeline, the epilogue WG can start the next block as
                // soon as scratch is released. Re-converge the math WG here and make
                // the shared-store visibility explicit before handing scratch over.
                warpgroup::sync(1);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        block_idx == debug_target_block &&
                        lane_id == 0) {
                        printf("DBG math_post_final_store_wait cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                               cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                    }
                }
                if (warpgroup::warpid() == 0 && lane_id == 0) {
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    arrive(epi_scratch_released, 1);
                    arrive(epi_finished, 1);
                    arrive_remote_cluster_v5(epi_finished, 1 - cta_id, 1);
                    if constexpr (Debug) {
                        if (g.debug_trace_mode > 0 &&
                            block_idx == debug_target_block) {
                            printf("DBG math_post_epi_arrive cta=%d warp=%d block=%d issue=%08x acc=%08x\n",
                                   cta_id, warpgroup::warpid(), block_idx, issue_phasebits, acc_phasebits);
                        }
                    }
                }
                warpgroup::sync(1);
                ++produced_blocks;

	            debug_trap_block(52, block_idx);
	            debug_trap_prev_block(62, block_idx);
        }

        // Drain the final phase-3 recycle before teardown. The steady-state path
        // consumes this at the next ksub or next block entry, but the last block
        // in a cluster would otherwise exit with the recycle token still live.
        if (p3_tm_live) {
            wait(p3_outputs_finished, get_phasebit<0>(issue_phasebits, 8));
            tensor_after_thread_sync();
            warpgroup::sync(1);
            update_phasebit<0>(issue_phasebits, 8);
            p3_tm_live = false;
        }
        if constexpr (Debug) {
            if (g.debug_breakpoint == 120 &&
                last_block_idx == debug_target_block &&
                warpgroup::warpid() == 0 &&
                lane_id == 0) {
                printf("DBG marker=120 cluster=%d cta=%d issue=%08x acc=%08x last_block=%d\n",
                       cluster_id, cta_id, issue_phasebits, acc_phasebits, last_block_idx);
                asm volatile("trap;");
            }
        }

        // Match the public dC teardown: drain any outstanding async store reads
        // before deprovisioning TMEM at warpgroup exit.
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        wait(epi_kernel_done, 0);
        warpgroup::sync(1);
        if constexpr (Debug) {
            if (g.debug_breakpoint == 121 &&
                last_block_idx == debug_target_block &&
                warpgroup::warpid() == 0 &&
                lane_id == 0) {
                printf("DBG marker=121 cluster=%d cta=%d issue=%08x acc=%08x last_block=%d\n",
                       cluster_id, cta_id, issue_phasebits, acc_phasebits, last_block_idx);
                asm volatile("trap;");
            }
        }
        if constexpr (Debug) {
            if (g.debug_trace_mode > 0 &&
                lane_id == 0) {
                printf("DBG math_post_exit_wait cta=%d warp=%d issue=%08x acc=%08x\n",
                       cta_id, warpgroup::warpid(), issue_phasebits, acc_phasebits);
            }
        }
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
            if constexpr (Debug) {
                if (g.debug_breakpoint == 122 &&
                    last_block_idx == debug_target_block &&
                    lane_id == 0) {
                    printf("DBG marker=122 cluster=%d cta=%d issue=%08x acc=%08x last_block=%d\n",
                           cluster_id, cta_id, issue_phasebits, acc_phasebits, last_block_idx);
                    asm volatile("trap;");
                }
            }
            if constexpr (Debug) {
                if (g.debug_trace_mode > 0 && lane_id == 0) {
                    printf("DBG math_post_deprovision cta=%d warp=%d issue=%08x acc=%08x\n",
                           cta_id, warpgroup::warpid(), issue_phasebits, acc_phasebits);
                }
            }
        }
    } else if (warpgroup_id == C::EPILOGUE_WARPGROUP_ID) {
        everyone::tma::cluster::wait_aligned();
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        auto phase1_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto phase1_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
        const float global_scale = g.A_sc_global[{0}] * g.B_sc_global[{0}];
        const int lane_id = threadIdx.x % 32;
        const int wg_thread = warpgroup::warpid() * WARP_THREADS + lane_id;
        constexpr int EPI_SYNC_BARRIER = 2;
        using logits_rt = rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        using logits_rt_bf = rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
        constexpr float kFp4Max = 6.0f;
        constexpr float kE4M3Max = 448.0f;
        const float gt_row_sg_val = fmaxf(g.grad_scale / (kFp4Max * kE4M3Max), 1.0e-12f);
        const float g_sg_rcp = 1.0f / gt_row_sg_val;
        int a_consumed_phase = 0;
        int a_clear_phase = 0;
        int scratch_release_phase = 0;
        bool have_completed_block = false;
        int epi_last_block_idx = -1;

        for (int block_idx = block_loop_start; block_idx < num_blocks; block_idx += block_loop_stride) {
            if (should_stop_block(block_idx)) break;
            epi_last_block_idx = block_idx;
            if (have_completed_block) {
                warpgroup::sync(EPI_SYNC_BARRIER);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0) {
                        debug_log_epi(
                            TRACE_EPI_PRE_SCRATCH_WAIT, block_idx, -1, -1,
                            scratch_release_phase, a_clear_phase);
                    }
                }
                wait(epi_scratch_released, scratch_release_phase);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0) {
                        debug_log_epi(
                            TRACE_EPI_POST_SCRATCH_WAIT, block_idx, -1, -1,
                            scratch_release_phase, a_clear_phase);
                    }
                }
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                scratch_release_phase ^= 1;
                warpgroup::sync(EPI_SYNC_BARRIER);
            }
            const int supergroup_idx = block_idx / num_blocks_per_supergroup;
            const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            const int vocab_superblocks_in_group = min(C::SUPERGROUP_SIZE, num_vocab_superblocks - supergroup_idx * C::SUPERGROUP_SIZE);
            const int vocab_superblock_within_group = idx_within_supergroup % vocab_superblocks_in_group;
            const int vocab_superblock_idx = supergroup_idx * C::SUPERGROUP_SIZE + vocab_superblock_within_group;
            (void)idx_within_supergroup;
            // Clear our local staged A tile once per row-block sweep iteration.
            auto clear_a_super = [&](int row_block_idx) {
                constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1) {
                        debug_log_epi(
                            TRACE_EPI_PRE_GT_ROW_CLEAR, block_idx, row_block_idx, -1,
                            a_clear_phase, static_cast<int>(sizeof(typename G::Gt_fp4_row_tile)));
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_pre_gt_row_clear cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                uint8_t *fp4_bytes = reinterpret_cast<uint8_t*>(&fp4_staging.Gt_row.data[0]);
                for (int idx = wg_thread; idx < (int)sizeof(typename G::Gt_fp4_row_tile); idx += CONSUMER_THREADS) {
                    fp4_bytes[idx] = 0;
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1) {
                        debug_log_epi(
                            TRACE_EPI_POST_GT_ROW_CLEAR, block_idx, row_block_idx, -1,
                            a_clear_phase, static_cast<int>(sizeof(typename G::Gt_fp4_row_tile)));
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_post_gt_row_clear cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1) {
                        debug_log_epi(
                            TRACE_EPI_PRE_GT_SC_CLEAR, block_idx, row_block_idx, -1,
                            a_clear_phase, static_cast<int>(sizeof(typename G::Gt_sc_row_tile)));
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_pre_gt_sc_clear cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                uint8_t *sc_bytes = reinterpret_cast<uint8_t*>(&fp4_staging.Gt_row_sc.data[0]);
                for (int idx = wg_thread; idx < (int)sizeof(typename G::Gt_sc_row_tile); idx += CONSUMER_THREADS) {
                    sc_bytes[idx] = 0;
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1) {
                        debug_log_epi(
                            TRACE_EPI_POST_GT_SC_CLEAR, block_idx, row_block_idx, -1,
                            a_clear_phase, static_cast<int>(sizeof(typename G::Gt_sc_row_tile)));
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_post_gt_sc_clear cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
            };

            for (int row_block_idx = 0; row_block_idx < num_row_blocks; ++row_block_idx) {
                if (should_stop_row_block(row_block_idx)) break;
                const bool wait_for_a_super = row_block_idx > 0;
                debug_trap(40, 0, block_idx, row_block_idx);
                debug_trap(10, 1, block_idx, row_block_idx);
                if (wait_for_a_super && warpgroup::warpid() == 0 && lane_id == 0) {
                    wait(a_super_consumed, a_consumed_phase);
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 && row_block_idx == 0) {
                        debug_log_epi(
                            TRACE_EPI_PRE_ROW_SYNC, block_idx, row_block_idx, -1,
                            wait_for_a_super ? 1 : 0, a_consumed_phase);
                    }
                }
                if (wait_for_a_super) warpgroup::sync(EPI_SYNC_BARRIER);
                if (wait_for_a_super) a_consumed_phase ^= 1;
                clear_a_super(row_block_idx);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 && row_block_idx == 0) {
                        debug_log_epi(
                            TRACE_EPI_PRE_CLEAR_SYNC, block_idx, row_block_idx, -1,
                            a_clear_phase, 0);
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_pre_clear_sync cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                warpgroup::sync(EPI_SYNC_BARRIER);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 && row_block_idx == 0) {
                        debug_log_epi(
                            TRACE_EPI_POST_CLEAR_SYNC, block_idx, row_block_idx, -1,
                            a_clear_phase, 0);
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_post_clear_sync cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                __threadfence_block();
                // This row-clear handoff feeds the local split math WG on the same CTA,
                // so publish it on both CTA and cluster async proxy paths.
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (warpgroup::warpid() == 0 && lane_id == 0) {
                    arrive(a_clear_ready, 1);
                    arrive_remote_cluster_v5(a_clear_ready, 1 - cta_id, 1);
                }
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 && row_block_idx == 0) {
                        debug_log_epi(
                            TRACE_EPI_PRE_CLEAR_WAIT, block_idx, row_block_idx, -1,
                            a_clear_phase, 0);
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_pre_clear_wait cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                wait(a_clear_ready, a_clear_phase);
                a_clear_phase ^= 1;
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 && row_block_idx == 0) {
                        debug_log_epi(
                            TRACE_EPI_POST_CLEAR_WAIT, block_idx, row_block_idx, -1,
                            a_clear_phase, 0);
                    }
                    if (g.debug_trace_mode > 0 &&
                        cta_id == 0 &&
                        block_idx == debug_target_block &&
                        row_block_idx <= 1 &&
                        lane_id == 0) {
                        printf("DBG epi_post_clear_wait cta=%d warp=%d row=%d a_clear_phase=%d\n",
                               cta_id, warpgroup::warpid(), row_block_idx, a_clear_phase);
                    }
                }
                warpgroup::sync(EPI_SYNC_BARRIER);
                if constexpr (Debug) {
                    if (g.debug_trace_mode > 0 && row_block_idx == 0) {
                        debug_log_epi(
                            TRACE_EPI_POST_ROW_SYNC, block_idx, row_block_idx, -1,
                            wait_for_a_super ? 1 : 0, a_consumed_phase);
                    }
                }
                debug_trap(41, 0, block_idx, row_block_idx);
                debug_trap(11, 1, block_idx, row_block_idx);

                for (int subpass = 0; subpass < 2; ++subpass) {
                    const int vocab_block_idx = vocab_superblock_idx * 2 + subpass;
                    const bool valid_subpass = vocab_block_idx < num_vocab_blocks;

                    if (valid_subpass) {
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 && row_block_idx == 0 && subpass == 0) {
                                debug_log_epi(
                                    TRACE_EPI_PRE_OUTPUTS_ARRIVED, block_idx, row_block_idx, subpass,
                                    get_phasebit<0>(phasebits, 9 + subpass), 0);
                            }
                            if (g.debug_trace_mode > 0 &&
                                cta_id == 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx <= 1 &&
                                subpass <= 1 &&
                                lane_id == 0) {
                                printf("DBG epi_pre_outputs_arrived cta=%d warp=%d row=%d subpass=%d phase=%d\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, subpass,
                                       get_phasebit<0>(phasebits, 9 + subpass));
                            }
                        }
                        wait(outputs_arrived[subpass], get_phasebit<0>(phasebits, 9 + subpass));
                        if constexpr (Debug) {
                            if (g.debug_trace_mode > 0 && row_block_idx == 0 && subpass == 0) {
                                debug_log_epi(
                                    TRACE_EPI_POST_OUTPUTS_ARRIVED, block_idx, row_block_idx, subpass,
                                    get_phasebit<0>(phasebits, 9 + subpass), 0);
                            }
                            if (g.debug_trace_mode > 0 &&
                                cta_id == 0 &&
                                block_idx == debug_target_block &&
                                row_block_idx <= 1 &&
                                subpass <= 1 &&
                                lane_id == 0) {
                                printf("DBG epi_post_outputs_arrived cta=%d warp=%d row=%d subpass=%d phase=%d\n",
                                       cta_id, warpgroup::warpid(), row_block_idx, subpass,
                                       get_phasebit<0>(phasebits, 9 + subpass));
                            }
                        }

                        const int tile_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
                        const int warp_row_base = tile_row_base + warpgroup::warpid() * (C::Mb / 8);
                        const uint32_t gt_fp4_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row.data[0]));
                        const uint32_t gt_sc_base  = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row_sc.data[0]));
                        constexpr int SUBTILE_COLS = C::Nb / C::EPI_PIPE_DEPTH;
                        constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                        constexpr int LOCAL_ROW16_BLOCKS = (C::Mb / 2) / 16;
                        constexpr int ROW16_BLOCKS_PER_PASS = CONSUMER_THREADS / SUBTILE_COLS;
                        static_assert(LOCAL_ROW16_BLOCKS % ROW16_BLOCKS_PER_PASS == 0);

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
                        auto &phase1_tm = (subpass == 0) ? phase1_tm_0 : phase1_tm_1;
                        warpgroup::load_async(
                            D_pipe[0],
                            phase1_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, 0));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(EPI_SYNC_BARRIER);
                        int cur_slot = 0;

                        #pragma unroll
                        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                            const int next_slot = cur_slot ^ 1;
                            if constexpr (C::EPI_PIPE_DEPTH > 1) {
                                if (epi + 1 < C::EPI_PIPE_DEPTH) {
                                    warpgroup::load_async(
                                        D_pipe[next_slot],
                                        phase1_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, (epi + 1) * (C::Nb / C::EPI_PIPE_DEPTH)));
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

                            warpgroup::sync(EPI_SYNC_BARRIER);

                            #pragma unroll
                            for (int i = 0; i < logits_rt::height; ++i) {
                                const int global_row_x = warp_row_base + i * 16 + lane_id / 4;
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
                            warpgroup::sync(EPI_SYNC_BARRIER);
                            warpgroup::store(output_tiles.D[0], D_bf);
                            warpgroup::sync(EPI_SYNC_BARRIER);

                            const uint32_t d_base = static_cast<uint32_t>(__cvta_generic_to_shared(&output_tiles.D[0].data[0]));
                            const int col_in_epi = wg_thread % SUBTILE_COLS;
                            const int row16_block_base = wg_thread / SUBTILE_COLS;
                            const int local_gc = epi * SUBTILE_COLS + col_in_epi;

                            #pragma unroll
                            for (int row16_pass = 0; row16_pass < LOCAL_ROW16_BLOCKS / ROW16_BLOCKS_PER_PASS; ++row16_pass) {
                                const int row16_block = row16_block_base + row16_pass * ROW16_BLOCKS_PER_PASS;
                                if (row16_block >= LOCAL_ROW16_BLOCKS) continue;

                                const int local_row_base = row16_block * 16;
                                const int global_row_base = tile_row_base + local_row_base;
                                float col_amax = 0.0f;
                                #pragma unroll
                                for (int r = 0; r < 16; ++r) {
                                    bf16 value_bf;
                                    move<bf16>::lds(value_bf, G::Out_sm_tile::idx(d_base, {local_row_base + r, col_in_epi}));
                                    if (global_row_base + r < g.M) {
                                        col_amax = fmaxf(col_amax, fabsf(__bfloat162float(value_bf)));
                                    }
                                }

                                const float col_scale = col_amax * (1.0f / 6.0f);
                                const float col_rcp = (col_amax > 0.0f) ? (6.0f / col_amax) : 0.0f;
                                const int local_row_pair_base = cta_id * (C::Mb / 4) + local_row_base / 2;

                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const int global_row = global_row_base + pair * 2;
                                    if (global_row >= g.M) continue;

                                    bf16 value0_bf;
                                    move<bf16>::lds(value0_bf, G::Out_sm_tile::idx(d_base, {local_row_base + pair * 2, col_in_epi}));
                                    const float v0 = __bfloat162float(value0_bf);
                                    float v1 = 0.0f;
                                    if (global_row + 1 < g.M) {
                                        bf16 value1_bf;
                                        move<bf16>::lds(value1_bf, G::Out_sm_tile::idx(d_base, {local_row_base + pair * 2 + 1, col_in_epi}));
                                        v1 = __bfloat162float(value1_bf);
                                    }

                                    const uint8_t fp4_pair = quantize_fp4_pair_v5(v0, v1, col_rcp);
                                    const uint32_t fp4_addr = G::Gt_fp4_row_tile::idx(
                                        gt_fp4_base, {local_gc, local_row_pair_base + pair});
                                    if (subpass == cta_id) store_b8_local_v5(fp4_addr, fp4_pair);
                                    else                  store_b8_cluster_v5(fp4_addr, subpass, fp4_pair);
                                }

                                const __nv_fp8_e4m3 csc = __nv_fp8_e4m3(col_scale * g_sg_rcp);
                                const int local_row_m = cta_id * (C::Mb / 2) + local_row_base;
                                const int m_kgroup = local_row_m / 64;
                                const int m_16_in_64 = (local_row_m / 16) % 4;
                                const int sr = local_gc % 32;
                                const int rr = (local_gc / 32) % 4;
                                const int byte_idx = m_kgroup * 512 + sr * 16 + rr * 4 + m_16_in_64;
                                const uint8_t sc_byte = *reinterpret_cast<const uint8_t*>(&csc);
                                if (subpass == cta_id) store_b8_local_v5(gt_sc_base + byte_idx, sc_byte);
                                else                  store_b8_cluster_v5(gt_sc_base + byte_idx, subpass, sc_byte);
                            }

                            warpgroup::sync(EPI_SYNC_BARRIER);

                            if constexpr (C::EPI_PIPE_DEPTH > 1) {
                                if (epi + 1 < C::EPI_PIPE_DEPTH) {
                                    tensor_load_wait();
                                    tensor_before_thread_sync();
                                    warpgroup::sync(EPI_SYNC_BARRIER);
                                    cur_slot = next_slot;
                                }
                            }
                        }
                    }

                    warpgroup::sync(EPI_SYNC_BARRIER);
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                    if (warpgroup::warpid() == 0 && lane_id == 0) {
                        warpgroup::tma::cluster::arrive(outputs_finished[subpass], 0, 1);
                        arrive(a_super_ready[subpass], 1);
                        arrive_remote_cluster_v5(a_super_ready[subpass], 1 - cta_id, 1);
                    }
                    update_phasebit<0>(phasebits, 9 + subpass);
                }
            }

            if constexpr (Debug) {
                if (g.debug_trace_mode > 0 &&
                    block_idx == debug_target_block &&
                    lane_id == 0) {
                    printf("DBG epi_block_done cta=%d warp=%d block=%d phase0=%u phase1=%u\n",
                           cta_id, warpgroup::warpid(), block_idx,
                           get_phasebit<0>(phasebits, 9),
                           get_phasebit<0>(phasebits, 10));
                }
                if (g.debug_gt_fp4_ptr && cluster_id == 0 && block_idx == debug_target_block) {
                    constexpr int CONSUMER_THREADS = WARPGROUP_WARPS * WARP_THREADS;
                    const uint32_t gt_fp4_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row.data[0]));
                    const uint32_t gt_sc_base = static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_staging.Gt_row_sc.data[0]));
                    for (int idx = wg_thread; idx < C::Nb * (C::Mb / 2); idx += CONSUMER_THREADS) {
                        const int row = idx / (C::Mb / 2);
                        const int col = idx % (C::Mb / 2);
                        uint32_t value_u32;
                        asm volatile("ld.shared.b8 %0, [%1];\n" : "=r"(value_u32) : "r"(G::Gt_fp4_row_tile::idx(gt_fp4_base, {row, col})));
                        g.debug_gt_fp4_ptr[(cta_id * C::Nb + row) * g.debug_gt_fp4_stride + col] = static_cast<uint8_t>(value_u32);
                    }
                    for (int idx = wg_thread; idx < (int)sizeof(typename G::Gt_sc_row_tile); idx += CONSUMER_THREADS) {
                        g.debug_gt_sc_ptr[cta_id * (int)sizeof(typename G::Gt_sc_row_tile) + idx] =
                            reinterpret_cast<const uint8_t*>(&fp4_staging.Gt_row_sc.data[0])[idx];
                    }
                }
            }
            warpgroup::sync(EPI_SYNC_BARRIER);
            __threadfence_block();
            if (warpgroup::warpid() == 0 && lane_id == 0) {
                arrive(epi_block_done, 1);
                arrive_remote_cluster_v5(epi_block_done, 1 - cta_id, 1);
            }
            warpgroup::sync(EPI_SYNC_BARRIER);
            have_completed_block = true;
        }
        warpgroup::sync(EPI_SYNC_BARRIER);
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
        if (warpgroup::warpid() == 0 && lane_id == 0) {
            arrive(epi_kernel_done, 1);
            arrive_remote_cluster_v5(epi_kernel_done, 1 - cta_id, 1);
            if constexpr (Debug) {
                if (g.debug_breakpoint == 132 &&
                    epi_last_block_idx == debug_target_block) {
                    printf("DBG marker=132 cluster=%d cta=%d last_block=%d\n",
                           cluster_id, cta_id, epi_last_block_idx);
                    asm volatile("trap;");
                }
            }
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C>& g) {
    kernel_impl<C, globals<C>, false>(g);
}

template <typename C>
__device__ inline void debug_kernel(const debug_globals<C>& g) {
    kernel_impl<C, debug_globals<C>, true>(g);
}

// Developer-only probe placeholder. The benchmark path does not rely on it.
template <typename C>
struct p3_probe_globals {
    const uint8_t *gt_fp4_ptr;
    const uint8_t *gt_sc_ptr;
    const uint8_t *p3_b_fp4_ptr;
    const uint8_t *p3_b_sc_ptr;
    float gt_sg;
    float p3_b_sg;
    bf16 *out_bf_ptr;
    float *out_raw_ptr;
    int gt_fp4_stride;
    int p3_b_fp4_stride;
    int out_stride;
    int out_raw_stride;
    int probe_mode;

    __host__ inline dim3 grid() const { return dim3(1); }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const { return 0; }
};

template <typename C>
__device__ inline void p3_probe_kernel(const p3_probe_globals<C>&) {}

} // namespace nvfp4_cce_backward_v5_dC_superk4_experimental
