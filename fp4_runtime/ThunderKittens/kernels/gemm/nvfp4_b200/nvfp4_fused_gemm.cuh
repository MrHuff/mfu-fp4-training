#pragma once
// ================================================================
// NVFP4 Fused Quantize + GEMM Kernel
// Accepts bf16 A input, quantizes on-the-fly to NVFP4, then GEMMs.
// B is pre-quantized NVFP4.
// D = quantize(A_bf16) × B_fp4^T
//
// Template flag USE_CTA_AMAX controls the global scale strategy:
//   false: constant SCALE_MAX = bf16_max/(6*448) — fast, no sync
//   true:  per-sub-tile amax via warpgroup reduction of the per-block
//          amaxes already computed during quantization. No pre-scan.
//          Running max tracked in SMEM for the epilogue.
//
// Architecture: 3 warpgroups (384 threads)
//   WG0 (consumer):  epilogue — TMEM → registers → TMA store to HBM
//   WG1 (quantizer): TMA-loads bf16 A sub-tiles → quantizes to FP4 + scales
//   WG2 (producer):  TMA-loads B FP4/scales → MMA orchestration
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_fused_gemm {

// Constant global scale (used when USE_CTA_AMAX=false)
static constexpr float SCALE_MAX_DEC = 65504.0f / (6.0f * 448.0f);
static constexpr float SCALE_MAX_ENC = 1.0f / SCALE_MAX_DEC;

// Quantization tile constants
static constexpr int QUANT_TILE_M = 128;
static constexpr int QUANT_TILE_N = 128;
static constexpr int K_BLOCK_SIZE = 16;

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI, bool _USE_CTA_AMAX = false, int _Mb = 256, bool _USE_PDL = true, int _CLUSTER_SIZE = 2, int _COL_TILES_PER_BLOCK = 1>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_Mb == 256, "Fused kernel only supports Mb=256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5);
    static_assert(_EPI_PIPE_DEPTH > 0);
    static_assert(_SUPERGROUP_SIZE > 0);
    static_assert(_NUM_D_TILES > 0);
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2);
    static_assert(_COL_TILES_PER_BLOCK == 1 || (_COL_TILES_PER_BLOCK == 2 && _Nb == 128),
                  "Dual-column fused backend currently supports Nb=128 only");

    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    static constexpr bool USE_PDL = _USE_PDL;
    static constexpr bool USE_CTA_AMAX = _USE_CTA_AMAX;
    static constexpr int COL_TILES_PER_BLOCK = _COL_TILES_PER_BLOCK;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int QUANTIZER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + QUANTIZER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;  // 384

    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr bool OVERLAP_EPI = _OVERLAP_EPI;

    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;
    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int BLOCK_N = Nb * COL_TILES_PER_BLOCK;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;

    static constexpr int QUANT_SUB_TILES = Kb / QUANT_TILE_N; // 2
};

template <typename C>
struct globals {
    using A_bf16_tile  = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_bf16_gl      = gl<bf16,      1,  1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,      1, -1, -1, 256, A_sc_tile>;
    using B_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,      1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,     1,  1,  1,  1>;
    using D_gl           = gl<bf16,      1,  1, -1, -1, D_tile>;

    A_bf16_gl      A_bf16;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;
    D_gl           D;

    D_gl           D_K;
    D_gl           D_V;
    int            q_dim;
    int            k_dim;
    bool           use_split_D;
    const float*   b_sg_per_tile;
    int            silu_dim;

    // Scratch GMEM buffers for quantized A reuse across column tiles.
    // Sized [M, K/2] and [M/128, K/64, 512] respectively.
    // Populated during the first col tile and read for subsequent col tiles.
    A_fp4x2_gl     A_scratch;
    A_sc_gl        A_sc_scratch;
    // Per-(row_block, K_stage) completion flags, zeroed before launch.
    uint32_t*      A_scratch_ready;  // [num_row_blocks_total * num_red_blocks]

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B[C::COL_TILES_PER_BLOCK];
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::COL_TILES_PER_BLOCK][C::B_SC_SIZE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };
    struct quant_buf_t {
        A_bf16_tile bf16_tile;
    };
    // Decoupled quantizer output — separate from pipeline input_tiles/input_scales.
    // Split into two structs matching quantize_subtile_* function signatures.
    struct quant_out_tiles_t {
        A_fp4x2_tile A;  // matches input_tiles_t.A
    };
    struct quant_out_scales_t {
        A_sc_tile A;     // matches input_scales_t.A
    };

    __host__ inline dim3 grid() const {
        // Full row×col parallelism — quantizer is decoupled with GMEM flags.
        int d_cols = use_split_D ? (q_dim + k_dim + D_V.cols()) : D.cols();
        int num_blocks = (D.rows() / C::Mb) * (d_cols / C::BLOCK_N);
        return dim3(min(num_blocks * C::CLUSTER_SIZE, num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int _dynamic_shared_memory = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                                               sizeof(outputs_t) +
                                               sizeof(quant_buf_t) + 1024 +
                                               sizeof(quant_out_tiles_t) +
                                               sizeof(quant_out_scales_t) + 1024 +
                                               64;
        static_assert(_dynamic_shared_memory <= MAX_SHARED_MEMORY - 1024);
        return _dynamic_shared_memory;
    }
};


// ================================================================
// Constant-scale quantization for one 128x128 bf16 sub-tile.
// Processes one 64-column half at a time so the fast path carries a
// smaller live register set than the CTA-amax variant.
// Called by all 128 threads of the quantizer warpgroup.
// ================================================================
template <typename G, typename OutTiles, typename OutScales>
__device__ inline void quantize_subtile_constant(
    typename G::quant_buf_t &quant_buf,
    OutTiles &out_tile,
    OutScales &out_scales,
    int sub_tile_idx
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;  // 4
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;                      // 8

    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &A_bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8)%NUM_K_BLOCKS_HALF + col_half*NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*K_BLOCK_SIZE + ((local_tid+j)*2)%K_BLOCK_SIZE;
                const int offset = (tile_row*QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }

    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            bf16_2 _amax = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                _amax = __hmax2(_amax, __habs2(A_bf16_reg[col_half][i][j]));
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(_amax.x, _amax.y));
        }
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            A_sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * SCALE_MAX_ENC);

        const uint32_t a_tile_smem_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(&out_tile.A.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(A_sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec*SCALE_MAX_DEC, 1e-12f);
            const int fp4_col_base = sub_tile_idx * (QUANT_TILE_N/2) +
                                     (k_block_idx + col_half*NUM_K_BLOCKS_HALF)*K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int fp4_col = fp4_col_base + ((local_tid+j)&7);
                const uint32_t swizzled_addr = decltype(out_tile.A)::idx(
                    a_tile_smem_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x)*s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y)*s_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t*>(&out_scales.A.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[1][0])));
    }
}

// ================================================================
// CTA-amax quantization for one 128x128 bf16 sub-tile.
// Keeps the two 64-column halves live together so it can reduce a
// single sub-tile max before deriving per-block scales.
// Called by all 128 threads of the quantizer warpgroup.
// ================================================================
template <typename G, typename OutTiles, typename OutScales>
__device__ inline void quantize_subtile_cta_amax(
    typename G::quant_buf_t &quant_buf,
    OutTiles &out_tile,
    OutScales &out_scales,
    int sub_tile_idx,
    int quant_sync_bar_id,
    float *running_amax,
    float *warp_max_buf
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;  // 4
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;                      // 8

    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &A_bf16_smem = quant_buf.bf16_tile;

    // Load from SMEM
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8)%NUM_K_BLOCKS_HALF + col_half*NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*K_BLOCK_SIZE + ((local_tid+j)*2)%K_BLOCK_SIZE;
                const int offset = (tile_row*QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }

    // Compute per-16-element block amaxes (both col halves)
    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            bf16_2 _amax = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                _amax = __hmax2(_amax, __habs2(A_bf16_reg[col_half][i][j]));
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(_amax.x, _amax.y));
        }
    }

    // Warpgroup-reduce all per-block amaxes to get sub-tile amax.
    float thread_max = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < 2; ch++)
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            thread_max = fmaxf(thread_max, amax_all[ch][i]);

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));

    const int warp_in_wg = local_tid / 32;
    const int lane = local_tid % 32;
    if (lane == 0) warp_max_buf[warp_in_wg] = thread_max;
    warpgroup::sync(quant_sync_bar_id);
    float subtile_amax = fmaxf(fmaxf(warp_max_buf[0], warp_max_buf[1]),
                               fmaxf(warp_max_buf[2], warp_max_buf[3]));
    warpgroup::sync(quant_sync_bar_id);

    subtile_amax = fmaxf(subtile_amax, 1e-12f);
    if (local_tid == 0) {
        float old = *running_amax;
        *running_amax = fmaxf(old, subtile_amax);
    }

    const float s_global_dec = subtile_amax / (6.0f * 448.0f);
    const float s_global_enc = (6.0f * 448.0f) / subtile_amax;

    // Compute FP8 block scales and quantize to FP4
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            A_sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * s_global_enc);

        const uint32_t a_tile_smem_base = static_cast<uint32_t>(
            __cvta_generic_to_shared(&out_tile.A.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + local_tid/8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(A_sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec*s_global_dec, 1e-12f);
            const int fp4_col_base = sub_tile_idx * (QUANT_TILE_N/2) +
                                     (k_block_idx + col_half*NUM_K_BLOCKS_HALF)*K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int fp4_col = fp4_col_base + ((local_tid+j)&7);
                const uint32_t swizzled_addr = decltype(out_tile.A)::idx(
                    a_tile_smem_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x)*s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y)*s_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t*>(&out_scales.A.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t*>(&A_sc_reg[1][0])));
    }
}


// SiLU activation
template <typename RT>
__device__ inline void apply_silu_inplace(RT &D_reg) {
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                auto &v = D_reg.tiles[i][j].data[k];
                v.x = v.x / (1.0f + __expf(-v.x));
                v.y = v.y / (1.0f + __expf(-v.y));
            }
        }
    }
}

// ================================================================
// Main kernel
// ================================================================
template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) printf("START KERNEL\n");
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
        if (g.use_split_D) {
            g.D_K.template prefetch_tma<typename G::D_tile>();
            g.D_V.template prefetch_tma<typename G::D_tile>();
        }
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_clusters = gridDim.x / C::CLUSTER_SIZE;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int N_total = g.use_split_D ? (g.q_dim + g.k_dim + g.D_V.cols()) : g.D.cols();
    const int num_col_blocks = N_total / C::BLOCK_N;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();
    typename G::quant_buf_t     &quant_buf                         = sm_allocator.allocate<G::quant_buf_t>();
    typename G::quant_out_tiles_t  &quant_out_tiles  = sm_allocator.allocate<G::quant_out_tiles_t>();
    typename G::quant_out_scales_t &quant_out_scales = sm_allocator.allocate<G::quant_out_scales_t>();

    // SMEM for CTA-amax mode
    __shared__ float running_amax;
    __shared__ float warp_max_buf[4];

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Barriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore bf16_sub_arrived[C::QUANT_SUB_TILES];
    // Barriers for A-from-scratch TMA loads (separate from cluster barriers)
    __shared__ semaphore a_scratch_arrived[2]; // [0] = tiles, [1] = scales

    // Supergroup iteration variables (used by producer/MMA/consumer)
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;

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
        for (int s = 0; s < C::QUANT_SUB_TILES; s++)
            init_semaphore(bf16_sub_arrived[s], 0, 1);
        init_semaphore(a_scratch_arrived[0], 0, 1);
        init_semaphore(a_scratch_arrived[1], 0, 1);
        if constexpr (C::USE_CTA_AMAX)
            running_amax = 0.0f;
    }
    everyone::tma::cluster::arrive_aligned();

    // ================================================================
    // WG1: Decoupled Quantizer — independent row-only loop
    //   Runs ahead of the MMA pipeline.  Writes quantized A to GMEM
    //   scratch and sets per-(row, K_stage) atomic completion flags.
    //   Does NOT participate in pipeline barriers (inputs_finished,
    //   tiles_arrived, etc.)  — those are producer-/MMA-only.
    // ================================================================
    if (warpgroup_id == C::CONSUMER_WARPGROUPS) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int quant_sync_bar_id = 2;

        if constexpr (C::USE_PDL) {
            if (warp::laneid() == 0) pdl::wait();
        }
        everyone::tma::cluster::wait();

        int bf16_sub_phase = 0;

        for (int row_block_idx = cluster_id; row_block_idx < num_row_blocks; row_block_idx += num_clusters) {
            if constexpr (C::USE_CTA_AMAX) {
                if (is_leader) running_amax = 0.0f;
                warpgroup::sync(quant_sync_bar_id);
            }

            for (int i = 0; i < num_red_blocks; ++i) {
                // Load bf16 A sub-tiles → quantize → write into quant_out SMEM
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; sub++) {
                    if (is_leader) {
                        tma::expect(bf16_sub_arrived[sub], quant_buf.bf16_tile);
                        tma::load_async(quant_buf.bf16_tile, g.A_bf16,
                            {row_block_idx*2 + cta_id, i * C::QUANT_SUB_TILES + sub},
                            bf16_sub_arrived[sub]);
                    }
                    wait(bf16_sub_arrived[sub], bf16_sub_phase);

                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_subtile_cta_amax<G>(
                            quant_buf, quant_out_tiles, quant_out_scales,
                            sub, quant_sync_bar_id, &running_amax, warp_max_buf);
                    } else {
                        quantize_subtile_constant<G>(
                            quant_buf, quant_out_tiles, quant_out_scales, sub);
                    }
                    warpgroup::sync(quant_sync_bar_id);
                }
                bf16_sub_phase ^= 1;

                // Fence SMEM writes visible for TMA store
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                // TMA store quantized A tiles + scales from quant_out → GMEM scratch
                if (is_leader) {
                    tma::store_async(g.A_scratch, quant_out_tiles.A,
                        {row_block_idx*2 + cta_id, i});
                    tma::store_async(g.A_sc_scratch, quant_out_scales.A,
                        {row_block_idx*2 + cta_id, i, 0});
                }
            // Wait for TMA stores to complete reading from SMEM
                if (is_leader) {
                    warpgroup::tma::store_commit_group();
                    warpgroup::tma::store_async_wait<0>();
                }
                warpgroup::sync(quant_sync_bar_id);

                // Global fence + proxy fence: make scratch visible to all CTAs
                asm volatile("fence.proxy.async.global;\n" ::: "memory");
                __threadfence_system();
                if (is_leader) {
                    const int flag_idx = (row_block_idx * 2 + cta_id) * num_red_blocks + i;
                    atomicExch(&g.A_scratch_ready[flag_idx], 1u);
                }
            }
        }
    }
    // ================================================================
    // WG2: Producer — TMA loads A (from scratch) + B, MMA
    // ================================================================
    else if (warpgroup_id == C::CONSUMER_WARPGROUPS + C::QUANTIZER_WARPGROUPS) {
        if (warp::elect_leader()) {
            int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
            if (warp_id == 3) {
                // ── Tile loader: poll GMEM flag, TMA load A from scratch + B ──
                if constexpr (C::USE_PDL) pdl::wait();
                everyone::tma::cluster::wait();
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += num_clusters) {
                    const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    const int idx_within = block_idx % num_blocks_per_supergroup;
                    const int rows_in_sg = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    const int col_block_idx = idx_within / rows_in_sg;
                    const int row_within = idx_within % rows_in_sg;
                    const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within;

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));

                        // Poll GMEM flag until A scratch is ready
                        {
                            const int flag_idx = (row_block_idx * 2 + cta_id) * num_red_blocks + i;
                            while (atomicAdd(&g.A_scratch_ready[flag_idx], 0u) == 0u) {}
                        }

                        // TMA load A from scratch — use dedicated barrier, wait for completion
                        tma::expect(a_scratch_arrived[0], input_tiles[stage].A);
                        tma::load_async(input_tiles[stage].A, g.A_scratch,
                            {row_block_idx*2 + cta_id, i},
                            a_scratch_arrived[0]);
                        wait(a_scratch_arrived[0], get_phasebit<0>(phasebits, 1));
                        update_phasebit<0>(phasebits, 1);

                        // TMA load B tiles (cluster barrier, as original)
                        if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                            tma::cluster::load_async(input_tiles[stage].B[0], g.B,
                                {col_block_idx*2 + cta_id, i},
                                tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                        } else {
                            #pragma unroll
                            for (int col_tile = 0; col_tile < C::COL_TILES_PER_BLOCK; ++col_tile) {
                                tma::cluster::load_async(input_tiles[stage].B[col_tile], g.B,
                                    {(col_block_idx*C::COL_TILES_PER_BLOCK + col_tile)*2 + cta_id, i},
                                    tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                            }
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            } else if (warp_id == 2) {
                // ── Scale loader: TMA load A scales from scratch + B scales ──
                if constexpr (C::USE_PDL) pdl::wait();
                everyone::tma::cluster::wait();
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += num_clusters) {
                    const int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    const int idx_within = block_idx % num_blocks_per_supergroup;
                    const int rows_in_sg = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    const int col_block_idx = idx_within / rows_in_sg;
                    const int row_within = idx_within % rows_in_sg;
                    const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within;

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));

                        // Poll GMEM flag until A scratch is ready
                        {
                            const int flag_idx = (row_block_idx * 2 + cta_id) * num_red_blocks + i;
                            while (atomicAdd(&g.A_scratch_ready[flag_idx], 0u) == 0u) {}
                        }

                        // TMA load A scales from scratch — use dedicated barrier, wait
                        tma::expect(a_scratch_arrived[1], input_scales[stage].A);
                        tma::load_async(input_scales[stage].A, g.A_sc_scratch,
                            {row_block_idx*2 + cta_id, i, 0},
                            a_scratch_arrived[1]);
                        wait(a_scratch_arrived[1], get_phasebit<0>(phasebits, 1));
                        update_phasebit<0>(phasebits, 1);

                        // TMA load B scales
                        if constexpr (C::COL_TILES_PER_BLOCK == 1 && C::B_SC_SIZE == 2) {
                            tma::cluster::load_async(input_scales[stage].B[0][cta_id], g.B_sc,
                                {col_block_idx*2 + cta_id, i, 0},
                                scales_arrived[stage], (uint16_t)(0b11), 0);
                        } else if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                            if (cta_id == 0) {
                                tma::cluster::load_async(input_scales[stage].B[0][0], g.B_sc,
                                    {col_block_idx, i, 0},
                                    scales_arrived[stage], (uint16_t)(0b11), 0);
                            }
                        } else if (cta_id == 0) {
                            #pragma unroll
                            for (int col_tile = 0; col_tile < C::COL_TILES_PER_BLOCK; ++col_tile) {
                                tma::cluster::load_async(input_scales[stage].B[col_tile][0], g.B_sc,
                                    {col_block_idx*C::COL_TILES_PER_BLOCK + col_tile, i, 0},
                                    scales_arrived[stage], (uint16_t)(0b11), 0);
                            }
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            } else if (cta_id == 0 && warp_id == 0) {
                // ── MMA orchestration (A loaded via TMA from scratch by producer) ──
                everyone::tma::cluster::wait();
                wait(tmem_provisioned, 0);
                tm_allocator.set_addr(tmem_addr);
                if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                    auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
                    auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

                    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += num_clusters) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        tensor_after_thread_sync();
                        for (int i = 0; i < num_red_blocks; i++) {
                            // Wait for A+B scales (loaded by warp 2)
                            tma::expect_bytes(scales_arrived[stage],
                                C::CLUSTER_SIZE * C::B_SC_SIZE * sizeof(typename G::B_sc_tile));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));

                            #pragma unroll
                            for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                                auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                                auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                                load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                                auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16);
                                auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][0].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                                if constexpr (C::B_SC_SIZE == 2) {
                                    auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16+16);
                                    auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][1].data[0])+16*32*ii);
                                    load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                                }
                            }

                            tma::expect_bytes(tiles_arrived[stage],
                                C::CLUSTER_SIZE * sizeof(typename G::B_fp4x2_tile));
                            wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));

                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                            if (i == 0) mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0],
                                                A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                                B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                                inputs_finished[stage]);
                            else       mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0],
                                                A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                                B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                                inputs_finished[stage]);
                            update_phasebit<0>(phasebits, stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                        tensor_commit<2>(outputs_arrived);
                        update_phasebit<1>(phasebits, 0);
                    }
                } else {
                    auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                    auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE>>(256);
                    auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE>>(256 + 4*C::MMA_PER_TILE);
                    auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE>>(256 + 8*C::MMA_PER_TILE);

                    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += num_clusters) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        tensor_after_thread_sync();
                        for (int i = 0; i < num_red_blocks; i++) {
                            tma::expect_bytes(scales_arrived[stage],
                                C::CLUSTER_SIZE * C::COL_TILES_PER_BLOCK * sizeof(typename G::B_sc_tile));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));

                            #pragma unroll
                            for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                                auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                                auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0])+16*32*ii);
                                load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);

                                auto B_sc_tm_subtile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                                auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0][0].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                                auto B_sc_tm_subtile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(ii*16);
                                auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1][0].data[0])+16*32*ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                            }

                            tma::expect_bytes(tiles_arrived[stage],
                                C::CLUSTER_SIZE * C::COL_TILES_PER_BLOCK * sizeof(typename G::B_fp4x2_tile));
                            wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));

                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                            if (i == 0) {
                                mm2_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0],
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                        B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0));
                                mm2_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1],
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                        B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                        inputs_finished[stage]);
                            } else {
                                mma2_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0],
                                         A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                         B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0));
                                mma2_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1],
                                         A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                         B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(0),
                                         inputs_finished[stage]);
                            }
                            update_phasebit<0>(phasebits, stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                        tensor_commit<2>(outputs_arrived);
                        update_phasebit<1>(phasebits, 0);
                    }
                }
            }
        }
    }
    // ================================================================
    // WG0: Consumer — epilogue
    // ================================================================
    else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);

        const float default_b_sg = g.B_sc_global[{0}];

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += num_clusters) {
            const int supergroup_idx = block_idx / num_blocks_per_supergroup;
            const int idx_within = block_idx % num_blocks_per_supergroup;
            const int rows_in_sg = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            const int col_block_idx = idx_within / rows_in_sg;
            const int row_within = idx_within % rows_in_sg;
            const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            float a_sg_dec;
            if constexpr (C::USE_CTA_AMAX) {
                a_sg_dec = running_amax / (6.0f * 448.0f);
            } else {
                a_sg_dec = SCALE_MAX_DEC;
            }

            if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

                if constexpr (C::OVERLAP_EPI) {
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg;
                        warpgroup::load_async(D_reg, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                        if (i == C::EPI_PIPE_DEPTH - 1) {
                            tensor_load_wait();
                            tensor_before_thread_sync();
                            warpgroup::sync(1);
                            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                        }
                        {
                            float gs = a_sg_dec * default_b_sg;
                            if (g.b_sg_per_tile != nullptr) {
                                gs = a_sg_dec * g.b_sg_per_tile[col_block_idx];
                            }
                            warp::mul(D_reg, D_reg, gs);
                        }
                        if (g.silu_dim > 0) {
                            int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.silu_dim) apply_silu_inplace(D_reg);
                        }

                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg);
                        warpgroup::sync(1);
                        if (g.use_split_D) {
                            int col_offset = C::EPI_PIPE_DEPTH*col_block_idx + i;
                            int col_offset_elems = col_offset * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.q_dim) {
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, col_offset});
                            } else if (col_offset_elems < g.q_dim + g.k_dim) {
                                int k_col_offset = col_offset - (g.q_dim / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_K, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, k_col_offset});
                            } else {
                                int v_col_offset = col_offset - ((g.q_dim + g.k_dim) / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_V, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, v_col_offset});
                            }
                        } else {
                            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
                        }
                    }
                } else {
                    rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                        warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                        {
                            float gs = a_sg_dec * default_b_sg;
                            if (g.b_sg_per_tile != nullptr) {
                                gs = a_sg_dec * g.b_sg_per_tile[col_block_idx];
                            }
                            warp::mul(D_reg_fl, D_reg_fl, gs);
                        }
                        if (g.silu_dim > 0) {
                            int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.silu_dim) apply_silu_inplace(D_reg_fl);
                        }
                        warp::copy(D_reg[i], D_reg_fl);
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                        warpgroup::sync(1);
                        if (g.use_split_D) {
                            int col_offset = C::EPI_PIPE_DEPTH*col_block_idx + i;
                            int col_offset_elems = col_offset * C::Nb/C::EPI_PIPE_DEPTH;
                            if (col_offset_elems < g.q_dim) {
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, col_offset});
                            } else if (col_offset_elems < g.q_dim + g.k_dim) {
                                int k_col_offset = col_offset - (g.q_dim / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_K, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, k_col_offset});
                            } else {
                                int v_col_offset = col_offset - ((g.q_dim + g.k_dim) / (C::Nb/C::EPI_PIPE_DEPTH));
                                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D_V, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, v_col_offset});
                            }
                        } else {
                            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
                        }
                    }
                }
            } else {
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

                auto store_dual_tile = [&](auto &out_tm_sel, int col_tile_offset, bool release_outputs) {
                    rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                        warpgroup::load_async(D_reg_fl, out_tm_sel.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                        warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * default_b_sg);
                        warp::copy(D_reg[i], D_reg_fl);
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    if (release_outputs) {
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    }
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                        warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                        warpgroup::sync(1);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                            g.D, output_tiles.D[i%C::NUM_D_TILES],
                            {row_block_idx*2 + cta_id,
                             C::EPI_PIPE_DEPTH*(col_block_idx*C::COL_TILES_PER_BLOCK + col_tile_offset) + i});
                    }
                };

                store_dual_tile(out_tm_0, 0, false);
                store_dual_tile(out_tm_1, 1, true);
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace nvfp4_fused_gemm
