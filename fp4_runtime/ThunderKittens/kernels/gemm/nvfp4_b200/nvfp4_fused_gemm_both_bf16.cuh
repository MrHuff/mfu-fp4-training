#pragma once
// ================================================================
// Experimental two-sided fused NVFP4 GEMM.
//
// Accepts bf16 A and bf16 B, quantizes both operands on the fly inside
// a single CTA, then performs NVFP4 MMA:
//   D = quantize(A_bf16) x quantize(B_bf16)^T
//
// This is intentionally separate from the production fused kernel so we
// can benchmark a real "both operands in-kernel" prototype without
// destabilizing the current A-only fused path.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_fused_gemm_both_bf16 {

static constexpr float SCALE_MAX_DEC = 65504.0f / (6.0f * 448.0f);
static constexpr float SCALE_MAX_ENC = 1.0f / SCALE_MAX_DEC;

#if defined(SHARED_B_DEBUG_STANDALONE)
#define SHARED_B_DEBUG_PRINTF(...) printf(__VA_ARGS__)
#else
#define SHARED_B_DEBUG_PRINTF(...) ((void)0)
#endif

static constexpr int QUANT_TILE_M = 128;
static constexpr int QUANT_TILE_N = 128;
static constexpr int K_BLOCK_SIZE = 16;

template <
    bool _USE_CTA_AMAX,
    int _COL_TILES_PER_BLOCK = 1,
    int _LOAD_PIPE_DEPTH = 2,
    int _EPI_PIPE_DEPTH = 4,
    int _NUM_D_TILES = (_EPI_PIPE_DEPTH > 1 ? 4 : 1),
    int _CLUSTER_SIZE = 1,
    bool _SHARE_A_ACROSS_CTAS = false,
    bool _BYPASS_SHARED_A_ON_CTA1 = false,
    bool _PUBLISH_SHARED_A_WHEN_BYPASSED = false,
    bool _DEDICATED_PRODUCER = false,
    bool _CLUSTER_REUSE_2D = false,
    bool _SHARE_B_ACROSS_CTAS = false,
    int _KB = 256,
    bool _PREQUANT_A = false>
struct config {
    static_assert(_COL_TILES_PER_BLOCK == 1 || _COL_TILES_PER_BLOCK == 2 || _COL_TILES_PER_BLOCK == 4,
                  "Both-bf16 fused kernel supports 1, 2, or 4 column tiles per block");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 4,
                  "Both-bf16 fused kernel supports load pipe depth 1..4");
    static_assert(_EPI_PIPE_DEPTH > 0 && 128 % _EPI_PIPE_DEPTH == 0,
                  "Both-bf16 fused kernel requires EPI_PIPE_DEPTH to divide Nb=128");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be positive");
    static_assert(!_SHARE_A_ACROSS_CTAS || (_CLUSTER_SIZE == 2 && _COL_TILES_PER_BLOCK == 1),
                  "Shared-A both-bf16 path expects 2-CTA clusters and one local column tile per CTA");
    static_assert(!_BYPASS_SHARED_A_ON_CTA1 || _SHARE_A_ACROSS_CTAS,
                  "CTA1 local-A bypass is only valid for shared-A configs");
    static_assert(!_PUBLISH_SHARED_A_WHEN_BYPASSED || (_SHARE_A_ACROSS_CTAS && _BYPASS_SHARED_A_ON_CTA1),
                  "Shared-A publish-only mode requires shared-A clustering with CTA1 local-A bypass");
    static_assert(!_CLUSTER_REUSE_2D || (_CLUSTER_SIZE == 4 && _COL_TILES_PER_BLOCK == 1),
                  "Clustered 2x2 reuse expects four-CTA clusters and one local output tile per CTA");
    static_assert(_KB == 128 || _KB == 256,
                  "Both-bf16 fused kernel supports Kb=128 or Kb=256");
    static_assert(!_SHARE_B_ACROSS_CTAS ||
                      (_CLUSTER_SIZE == 2 && _COL_TILES_PER_BLOCK == 1 && _DEDICATED_PRODUCER && _KB == 128),
                  "Shared-B both-bf16 path expects a 2-CTA dedicated-producer backend with Kb=128");
    static_assert(!_PREQUANT_A || _SHARE_B_ACROSS_CTAS,
                  "Pre-quantized A mirror path currently reuses the shared-B 2CTA backend only");
    static constexpr bool USE_CTA_AMAX = _USE_CTA_AMAX;
    static constexpr int COL_TILES_PER_BLOCK = _COL_TILES_PER_BLOCK;
    static constexpr int CLUSTER_SIZE = _CLUSTER_SIZE;
    static constexpr bool SHARE_A_ACROSS_CTAS = _SHARE_A_ACROSS_CTAS;
    static constexpr bool BYPASS_SHARED_A_ON_CTA1 = _BYPASS_SHARED_A_ON_CTA1;
    static constexpr bool PUBLISH_SHARED_A_WHEN_BYPASSED = _PUBLISH_SHARED_A_WHEN_BYPASSED;
    static constexpr bool DEDICATED_PRODUCER = _DEDICATED_PRODUCER;
    static constexpr bool CLUSTER_REUSE_2D = _CLUSTER_REUSE_2D;
    static constexpr bool SHARE_B_ACROSS_CTAS = _SHARE_B_ACROSS_CTAS;
    static constexpr bool PREQUANT_A = _PREQUANT_A;
    static constexpr int TMEM_NCTA =
        (SHARE_A_ACROSS_CTAS || CLUSTER_REUSE_2D) ? 1 : (SHARE_B_ACROSS_CTAS ? 2 : CLUSTER_SIZE);

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int A_QUANT_WARPGROUPS = 1;
    static constexpr int B_QUANT_WARPGROUPS = 1;
    static constexpr int MMA_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS =
        CONSUMER_WARPGROUPS + A_QUANT_WARPGROUPS + B_QUANT_WARPGROUPS + MMA_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int Mb = 128;
    static constexpr int Nb = SHARE_B_ACROSS_CTAS ? 256 : 128;
    static constexpr int Kb = _KB;
    static constexpr int LOAD_PIPE_DEPTH = _LOAD_PIPE_DEPTH;
    static constexpr int QUANT_SUB_TILES = Kb / QUANT_TILE_N;
    static constexpr int MMA_PER_TILE = Kb / 64;

    static constexpr int EPI_PIPE_DEPTH = _EPI_PIPE_DEPTH;
    static constexpr int NUM_D_TILES = _NUM_D_TILES;
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;
};

template <typename C>
struct globals {
    using A_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using B_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb, C::Kb / 2>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb / (C::SHARE_B_ACROSS_CTAS ? C::CLUSTER_SIZE : 1), C::Kb / 2>;
    using A_sc_tile = st_hf<4, C::Kb, false>;
    using B_sc_tile = st_hf<4, C::Kb, false>;
    using D_tile = st_bf<C::Mb, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using B_bf16_gl = gl<bf16, 1, 1, -1, -1, B_bf16_tile>;
    using A_pre_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, A_fp4x2_tile>;
    using A_pre_sc_gl = gl<half, 1, -1, -1, 256, A_sc_tile>;
    using A_pre_sg_gl = gl<float, 1, 1, 1, 1>;
    using D_gl = gl<bf16, 1, 1, -1, -1, D_tile>;

    A_bf16_gl A_bf16;
    B_bf16_gl B_bf16;
    A_pre_fp4x2_gl A_pre;
    A_pre_sc_gl A_pre_sc;
    A_pre_sg_gl A_pre_sg;
    D_gl D;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B[C::COL_TILES_PER_BLOCK];
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::SHARE_B_ACROSS_CTAS ? C::CLUSTER_SIZE : C::COL_TILES_PER_BLOCK];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };
    struct quant_buf_t {
        A_bf16_tile bf16_tile;
    };
    struct a_export_t {
        alignas(16) uint8_t data[C::Mb * (C::Kb / 2)];
    };
    struct a_scale_export_t {
        alignas(16) uint8_t data[sizeof(A_sc_tile)];
    };
    using b_export_t = a_export_t;
    using b_scale_export_t = a_scale_export_t;

    __host__ inline dim3 grid() const {
        if constexpr (C::CLUSTER_REUSE_2D) {
            const int cluster_rows = D.rows() / (2 * C::Mb);
            const int cluster_cols = D.cols() / (2 * C::Nb);
            return dim3(cluster_rows * cluster_cols * C::CLUSTER_SIZE, 1, 1);
        }
        if constexpr (C::SHARE_B_ACROSS_CTAS) {
            return dim3((D.cols() / C::Nb) * C::CLUSTER_SIZE, D.rows() / (C::Mb * C::CLUSTER_SIZE));
        }
        if constexpr (C::SHARE_A_ACROSS_CTAS) {
            return dim3(D.cols() / C::Nb, D.rows() / C::Mb);
        }
        return dim3(D.cols() / (C::Nb * C::COL_TILES_PER_BLOCK), D.rows() / C::Mb);
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int smem =
            C::SHARE_B_ACROSS_CTAS
                ? (sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                   sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                   sizeof(outputs_t) + 1024 +
                   sizeof(A_bf16_tile) * C::LOAD_PIPE_DEPTH + 1024 +
                   sizeof(B_bf16_tile) * C::LOAD_PIPE_DEPTH + 1024 +
                   4096)
                : (sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                   sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                   sizeof(outputs_t) +
                   2 * sizeof(quant_buf_t) + 1024 +
                   (C::SHARE_A_ACROSS_CTAS && (!C::BYPASS_SHARED_A_ON_CTA1 || C::PUBLISH_SHARED_A_WHEN_BYPASSED)
                        ? C::LOAD_PIPE_DEPTH * (sizeof(a_export_t) + sizeof(a_scale_export_t)) + 1024
                        : 0));
        static_assert(smem <= MAX_SHARED_MEMORY - 1024);
        return smem;
    }

    uint8_t* debug_cta0_a_ptr;
    uint8_t* debug_cta1_a_ptr;
    uint8_t* debug_cta0_sc_ptr;
    uint8_t* debug_cta1_sc_ptr;
    int      debug_a_stride;
    bool     debug_transport_only;
    bool     debug_main_dump_only;
    int      debug_front_half_mode;
};

template <bool _USE_CTA_AMAX, bool _BYPASS_SHARED_A_ON_CTA1 = false, bool _DUMP_IMPORTED_TILE = false>
struct transport_debug_config {
    static constexpr bool USE_CTA_AMAX = _USE_CTA_AMAX;
    static constexpr bool BYPASS_SHARED_A_ON_CTA1 = _BYPASS_SHARED_A_ON_CTA1;
    static constexpr bool DUMP_IMPORTED_TILE = _DUMP_IMPORTED_TILE;
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_THREADS = 128;
    static constexpr int Mb = 128;
    static constexpr int Kb = 256;
    static constexpr int QUANT_SUB_TILES = Kb / QUANT_TILE_N;
};

template <typename C>
struct transport_debug_globals {
    using A_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb, C::Kb / 2>;
    using A_sc_tile = st_hf<4, 256, false>;
    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;

    struct local_tiles_t {
        A_fp4x2_tile A;
        A_sc_tile sc;
    };
    struct quant_buf_t {
        A_bf16_tile bf16_tile;
    };
    struct a_export_t {
        alignas(16) uint8_t data[C::Mb * (C::Kb / 2)];
    };

    A_bf16_gl A_bf16;
    uint8_t *debug_cta0_a_ptr;
    uint8_t *debug_cta1_a_ptr;
    uint8_t *debug_cta0_sc_ptr;
    uint8_t *debug_cta1_sc_ptr;
    int debug_a_stride;

    __host__ inline dim3 grid() const { return dim3(C::CLUSTER_SIZE, 1, 1); }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS, 1, 1); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int smem =
            sizeof(local_tiles_t) +
            sizeof(quant_buf_t) +
            1024;
        static_assert(smem <= MAX_SHARED_MEMORY - 1024);
        return smem;
    }
};

template <bool _USE_CTA_AMAX>
struct clustered_transport_debug_config {
    static constexpr bool USE_CTA_AMAX = _USE_CTA_AMAX;
    static constexpr int CLUSTER_SIZE = 4;
    static constexpr int NUM_THREADS = 128;
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Kb = 256;
    static constexpr int QUANT_SUB_TILES = Kb / QUANT_TILE_N;
};

template <typename C>
struct clustered_transport_debug_globals {
    using A_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using B_bf16_tile = st_bf<QUANT_TILE_M, QUANT_TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb, C::Kb / 2>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb, C::Kb / 2>;
    using A_sc_tile = st_hf<4, 256, false>;
    using B_sc_tile = st_hf<4, 256, false>;
    using A_bf16_gl = gl<bf16, 1, 1, -1, -1, A_bf16_tile>;
    using B_bf16_gl = gl<bf16, 1, 1, -1, -1, B_bf16_tile>;

    struct local_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
        A_sc_tile A_sc;
        B_sc_tile B_sc;
    };
    struct quant_buf_t {
        A_bf16_tile bf16_tile;
    };
    struct export_t {
        alignas(16) uint8_t data[C::Mb * (C::Kb / 2)];
    };
    struct scale_export_t {
        alignas(16) uint8_t data[sizeof(A_sc_tile)];
    };

    A_bf16_gl A_bf16;
    B_bf16_gl B_bf16;
    uint8_t *debug_a_owner_ptr;
    uint8_t *debug_a_recv_ptr;
    uint8_t *debug_a_owner_sc_ptr;
    uint8_t *debug_a_recv_sc_ptr;
    uint8_t *debug_b_owner_ptr;
    uint8_t *debug_b_recv_ptr;
    uint8_t *debug_b_owner_sc_ptr;
    uint8_t *debug_b_recv_sc_ptr;
    int debug_stride;

    __host__ inline dim3 grid() const { return dim3(C::CLUSTER_SIZE, 1, 1); }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS, 1, 1); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int smem =
            sizeof(local_tiles_t) +
            2 * sizeof(quant_buf_t) +
            1024;
        static_assert(smem <= MAX_SHARED_MEMORY - 1024);
        return smem;
    }
};

template <typename QuantBuf, typename Fp4Tile, typename ScTile>
__device__ inline void quantize_operand_subtile_constant(
    QuantBuf &quant_buf,
    Fp4Tile &out_tile,
    ScTile &out_scales,
    int sub_tile_idx,
    uint32_t export_canonical_smem_base = 0,
    uint32_t remote_swizzled_tile_local_smem_base = 0,
    uint32_t remote_canonical_smem_base = 0,
    uint32_t remote_scale_local_smem_base = 0,
    int remote_target_cta = -1,
    bool mirror_remote = false
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    bf16_2 bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF + col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col = k_block_idx * K_BLOCK_SIZE + ((local_tid + j) * 2) % K_BLOCK_SIZE;
                const int offset = (tile_row * QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_smem)) + offset);
            }
        }
    }

    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            bf16_2 amax_pair = __habs2(bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                amax_pair = __hmax2(amax_pair, __habs2(bf16_reg[col_half][i][j]));
            }
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(amax_pair.x, amax_pair.y));
        }
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * SCALE_MAX_ENC);
        }

        const uint32_t tile_base = static_cast<uint32_t>(__cvta_generic_to_shared(&out_tile.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec * SCALE_MAX_DEC, 1e-12f);
            const int fp4_col_base =
                sub_tile_idx * (QUANT_TILE_N / 2) +
                (k_block_idx + col_half * NUM_K_BLOCKS_HALF) * K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int fp4_col = fp4_col_base + ((local_tid + j) & 7);
                const uint32_t swizzled_addr = Fp4Tile::idx(tile_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(bf16_reg[col_half][i][j].x) * s_enc,
                    __bfloat162float(bf16_reg[col_half][i][j].y) * s_enc
                };
                const uint32_t packed_fp4 =
                    static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(packed_fp4));
                if (export_canonical_smem_base != 0) {
                    asm volatile("{st.shared.b8 [%0], %1;}"
                        :: "r"(export_canonical_smem_base + tile_row * QUANT_TILE_N + fp4_col),
                           "r"(packed_fp4));
                }
                if (mirror_remote && remote_canonical_smem_base != 0) {
                    nvfp4_fused_gemm::st_shared_cluster_b8(
                        nvfp4_fused_gemm::map_cluster_addr(
                            remote_canonical_smem_base + tile_row * QUANT_TILE_N + fp4_col,
                            remote_target_cta),
                        packed_fp4);
                }
                if (mirror_remote && remote_swizzled_tile_local_smem_base != 0) {
                    const uint32_t remote_swizzled_addr = Fp4Tile::idx(
                        remote_swizzled_tile_local_smem_base, {tile_row, fp4_col});
                    nvfp4_fused_gemm::st_shared_cluster_b8(
                        nvfp4_fused_gemm::map_cluster_addr(remote_swizzled_addr, remote_target_cta),
                        packed_fp4);
                }
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t *>(&out_scales.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[1][0])));
        if (mirror_remote && remote_scale_local_smem_base != 0) {
            nvfp4_fused_gemm::st_shared_cluster_b32(
                nvfp4_fused_gemm::map_cluster_addr(remote_scale_local_smem_base + mma_base_0 + scale_offset, remote_target_cta),
                *reinterpret_cast<uint32_t *>(&sc_reg[0][0]));
            nvfp4_fused_gemm::st_shared_cluster_b32(
                nvfp4_fused_gemm::map_cluster_addr(remote_scale_local_smem_base + mma_base_1 + scale_offset, remote_target_cta),
                *reinterpret_cast<uint32_t *>(&sc_reg[1][0]));
        }
    }
}

template <typename QuantBuf>
__device__ inline float compute_operand_subtile_amax(
    QuantBuf &quant_buf,
    int sub_tile_idx,
    int quant_sync_bar_id,
    float *warp_max_buf
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    bf16_2 bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    auto &bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF + col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col = sub_tile_idx * QUANT_TILE_N + k_block_idx * K_BLOCK_SIZE +
                                     ((local_tid + j) * 2) % K_BLOCK_SIZE;
                const int offset = (tile_row * QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_smem)) + offset);
            }
        }
    }

    float thread_max = 0.0f;
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            bf16_2 amax_pair = __habs2(bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                amax_pair = __hmax2(amax_pair, __habs2(bf16_reg[col_half][i][j]));
            }
            thread_max = fmaxf(thread_max, __bfloat162float(__hmax(amax_pair.x, amax_pair.y)));
        }
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    const int warp_in_wg = local_tid / 32;
    const int lane = local_tid % 32;
    if (lane == 0) {
        warp_max_buf[warp_in_wg] = thread_max;
    }
    warpgroup::sync(quant_sync_bar_id);
    const float subtile_amax = fmaxf(
        fmaxf(warp_max_buf[0], warp_max_buf[1]),
        fmaxf(warp_max_buf[2], warp_max_buf[3]));
    warpgroup::sync(quant_sync_bar_id);
    return subtile_amax;
}

template <typename QuantBuf, typename Fp4Tile, typename ScTile>
__device__ inline void quantize_operand_subtile_cta_amax_with_global(
    QuantBuf &quant_buf,
    Fp4Tile &out_tile,
    ScTile &out_scales,
    int sub_tile_idx,
    float global_subtile_amax,
    float *running_amax = nullptr
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    bf16_2 bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 sc_reg[2][NUM_K_BLOCKS_HALF];
    float amax_all[2][NUM_K_BLOCKS_HALF];

    auto &bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF + col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col = sub_tile_idx * QUANT_TILE_N + k_block_idx * K_BLOCK_SIZE +
                                     ((local_tid + j) * 2) % K_BLOCK_SIZE;
                const int offset = (tile_row * QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_smem)) + offset);
            }
        }
    }

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            bf16_2 amax_pair = __habs2(bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                amax_pair = __hmax2(amax_pair, __habs2(bf16_reg[col_half][i][j]));
            }
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(amax_pair.x, amax_pair.y));
        }
    }

    if (running_amax != nullptr && local_tid == 0) {
        *running_amax = fmaxf(*running_amax, fmaxf(global_subtile_amax, 1e-12f));
    }

    const float s_global_dec = fmaxf(global_subtile_amax, 1e-12f) / (6.0f * 448.0f);
    const float s_global_enc = (6.0f * 448.0f) / fmaxf(global_subtile_amax, 1e-12f);

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * s_global_enc);
        }

        const uint32_t tile_base = static_cast<uint32_t>(__cvta_generic_to_shared(&out_tile.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec * s_global_dec, 1e-12f);
            const int fp4_col_base =
                sub_tile_idx * (QUANT_TILE_N / 2) +
                (k_block_idx + col_half * NUM_K_BLOCKS_HALF) * K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int fp4_col = fp4_col_base + ((local_tid + j) & 7);
                const uint32_t swizzled_addr = Fp4Tile::idx(tile_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(bf16_reg[col_half][i][j].x) * s_enc,
                    __bfloat162float(bf16_reg[col_half][i][j].y) * s_enc
                };
                const uint32_t packed_fp4 =
                    static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr), "r"(packed_fp4));
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t *>(&out_scales.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[1][0])));
    }
}

template <typename QuantBuf, typename Fp4Tile, typename ScTile>
__device__ inline void quantize_operand_subtile_cta_amax(
    QuantBuf &quant_buf,
    Fp4Tile &out_tile,
    ScTile &out_scales,
    int sub_tile_idx,
    int quant_sync_bar_id,
    float *running_amax,
    float *warp_max_buf,
    uint32_t export_canonical_smem_base = 0,
    uint32_t remote_swizzled_tile_local_smem_base = 0,
    uint32_t remote_canonical_smem_base = 0,
    uint32_t remote_scale_local_smem_base = 0,
    int remote_target_cta = -1,
    bool mirror_remote = false
) {
    const int local_tid = threadIdx.x % 128;
    const int tile_row = local_tid;

    constexpr int NUM_K_BLOCKS_HALF = QUANT_TILE_N / K_BLOCK_SIZE / 2;
    constexpr int N_PER_K_BLOCK = K_BLOCK_SIZE / 2;

    bf16_2 bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK];
    fp8e4m3 sc_reg[2][NUM_K_BLOCKS_HALF];

    auto &bf16_smem = quant_buf.bf16_tile;

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF + col_half * NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int tile_col = k_block_idx * K_BLOCK_SIZE + ((local_tid + j) * 2) % K_BLOCK_SIZE;
                const int offset = (tile_row * QUANT_TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(
                    bf16_reg[col_half][i][j],
                    static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_smem)) + offset);
            }
        }
    }

    float amax_all[2][NUM_K_BLOCKS_HALF];
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            bf16_2 amax_pair = __habs2(bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; ++j) {
                amax_pair = __hmax2(amax_pair, __habs2(bf16_reg[col_half][i][j]));
            }
            amax_all[col_half][k_block_idx] = __bfloat162float(__hmax(amax_pair.x, amax_pair.y));
        }
    }

    float thread_max = 0.0f;
    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            thread_max = fmaxf(thread_max, amax_all[col_half][i]);
        }
    }

    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask));
    }

    const int warp_in_wg = local_tid / 32;
    const int lane = local_tid % 32;
    if (lane == 0) {
        warp_max_buf[warp_in_wg] = thread_max;
    }
    warpgroup::sync(quant_sync_bar_id);
    const float subtile_amax = fmaxf(
        fmaxf(warp_max_buf[0], warp_max_buf[1]),
        fmaxf(warp_max_buf[2], warp_max_buf[3]));
    warpgroup::sync(quant_sync_bar_id);

    if (local_tid == 0) {
        *running_amax = fmaxf(*running_amax, fmaxf(subtile_amax, 1e-12f));
    }

    const float s_global_dec = fmaxf(subtile_amax, 1e-12f) / (6.0f * 448.0f);
    const float s_global_enc = (6.0f * 448.0f) / fmaxf(subtile_amax, 1e-12f);

    #pragma unroll
    for (int col_half = 0; col_half < 2; ++col_half) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            sc_reg[col_half][i] = __nv_fp8_e4m3(amax_all[col_half][i] / 6.0f * s_global_enc);
        }

        const uint32_t tile_base = static_cast<uint32_t>(__cvta_generic_to_shared(&out_tile.data[0]));
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; ++i) {
            const int k_block_idx = (i + local_tid / 8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(sc_reg[col_half][k_block_idx]);
            const float s_enc = 1.0f / fmaxf(s_local_dec * s_global_dec, 1e-12f);
            const int fp4_col_base =
                sub_tile_idx * (QUANT_TILE_N / 2) +
                (k_block_idx + col_half * NUM_K_BLOCKS_HALF) * K_BLOCK_SIZE / 2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; ++j) {
                const int fp4_col = fp4_col_base + ((local_tid + j) & 7);
                const uint32_t swizzled_addr = Fp4Tile::idx(tile_base, {tile_row, fp4_col});
                const float2 scaled = {
                    __bfloat162float(bf16_reg[col_half][i][j].x) * s_enc,
                    __bfloat162float(bf16_reg[col_half][i][j].y) * s_enc
                };
                const uint32_t packed_fp4 =
                    static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest));
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(swizzled_addr),
                       "r"(packed_fp4));
                if (export_canonical_smem_base != 0) {
                    asm volatile("{st.shared.b8 [%0], %1;}"
                        :: "r"(export_canonical_smem_base + tile_row * QUANT_TILE_N + fp4_col),
                           "r"(packed_fp4));
                }
                if (mirror_remote && remote_canonical_smem_base != 0) {
                    nvfp4_fused_gemm::st_shared_cluster_b8(
                        nvfp4_fused_gemm::map_cluster_addr(
                            remote_canonical_smem_base + tile_row * QUANT_TILE_N + fp4_col,
                            remote_target_cta),
                        packed_fp4);
                }
                if (mirror_remote && remote_swizzled_tile_local_smem_base != 0) {
                    const uint32_t remote_swizzled_addr = Fp4Tile::idx(
                        remote_swizzled_tile_local_smem_base, {tile_row, fp4_col});
                    nvfp4_fused_gemm::st_shared_cluster_b8(
                        nvfp4_fused_gemm::map_cluster_addr(remote_swizzled_addr, remote_target_cta),
                        packed_fp4);
                }
            }
        }
    }

    {
        uint8_t *sc_base = reinterpret_cast<uint8_t *>(&out_scales.data[0]);
        const int mma_base_0 = sub_tile_idx * 2 * 512;
        const int mma_base_1 = mma_base_0 + 512;
        const int scale_offset = (tile_row % 32) * 16 + (tile_row / 32) * 4;

        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_0)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[0][0])));
        asm volatile("{st.shared.b32 [%0], %1;}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(sc_base + mma_base_1)) + scale_offset),
               "r"(*reinterpret_cast<uint32_t *>(&sc_reg[1][0])));
        if (mirror_remote && remote_scale_local_smem_base != 0) {
            nvfp4_fused_gemm::st_shared_cluster_b32(
                nvfp4_fused_gemm::map_cluster_addr(remote_scale_local_smem_base + mma_base_0 + scale_offset, remote_target_cta),
                *reinterpret_cast<uint32_t *>(&sc_reg[0][0]));
            nvfp4_fused_gemm::st_shared_cluster_b32(
                nvfp4_fused_gemm::map_cluster_addr(remote_scale_local_smem_base + mma_base_1 + scale_offset, remote_target_cta),
                *reinterpret_cast<uint32_t *>(&sc_reg[1][0]));
        }
    }
}

__device__ inline void store_shared_cluster_bytes_async_v4_b32(
    uint32_t dst_local_addr,
    uint32_t src_local_addr,
    int dst_cta,
    semaphore &remote_bar,
    int bytes,
    int thread_linear_idx,
    int thread_count
) {
    const uint32_t remote_bar_addr = nvfp4_fused_gemm::map_cluster_addr(
        static_cast<uint32_t>(__cvta_generic_to_shared(&remote_bar)), dst_cta);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int byte_off = thread_linear_idx * 16; byte_off < bytes; byte_off += thread_count * 16) {
        const uint32_t v0 = nvfp4_fused_gemm::ld_shared_u32(src_local_addr + byte_off + 0);
        const uint32_t v1 = nvfp4_fused_gemm::ld_shared_u32(src_local_addr + byte_off + 4);
        const uint32_t v2 = nvfp4_fused_gemm::ld_shared_u32(src_local_addr + byte_off + 8);
        const uint32_t v3 = nvfp4_fused_gemm::ld_shared_u32(src_local_addr + byte_off + 12);
        const uint32_t remote_dst_addr = nvfp4_fused_gemm::map_cluster_addr(dst_local_addr + byte_off, dst_cta);
        asm volatile(
            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n"
            :
            : "r"(remote_dst_addr),
              "r"(v0),
              "r"(v1),
              "r"(v2),
              "r"(v3),
              "r"(remote_bar_addr)
            : "memory");
    }
}

template <typename C>
__device__ inline void transport_debug_kernel(const transport_debug_globals<C> &g) {
    using G = transport_debug_globals<C>;
    static_assert(C::CLUSTER_SIZE == 2, "transport debug kernel expects a 2-CTA cluster");
    static_assert(C::NUM_THREADS == 128, "transport debug kernel expects one warpgroup per CTA");

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
    }

    const int cta_id = cluster_ctarank();
    const int local_tid = threadIdx.x;
    const bool is_leader = (local_tid == 0);
    constexpr int quant_sync_bar_id = 0;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::local_tiles_t &local_tiles = sm_allocator.allocate<typename G::local_tiles_t>();
    typename G::quant_buf_t &quant_buf = sm_allocator.allocate<typename G::quant_buf_t>();

    __shared__ typename G::a_export_t a_export_fp4;
    alignas(16) __shared__ uint8_t a_export_sc[sizeof(typename G::A_sc_tile)];
    __shared__ typename G::a_export_t a_recv_fp4;
    alignas(16) __shared__ uint8_t a_recv_sc[sizeof(typename G::A_sc_tile)];
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore a_payload_arrived;
    __shared__ float running_amax_a;
    __shared__ float warp_max_buf_a[4];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
        }
        init_semaphore(a_payload_arrived, 0, 1);
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
        }
    }

    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait();
    __syncthreads();

    if constexpr (!C::BYPASS_SHARED_A_ON_CTA1) {
        if (cta_id == 1 && is_leader) {
            tma::expect_bytes(a_payload_arrived, sizeof(typename G::a_export_t));
        }
        everyone::tma::cluster::sync();
    }

    int sub_phase = 0;
    if constexpr (C::USE_CTA_AMAX) {
        if (is_leader) {
            running_amax_a = 0.0f;
        }
        warpgroup::sync(quant_sync_bar_id);
    }

    if (cta_id == 0 || C::BYPASS_SHARED_A_ON_CTA1) {
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            if (is_leader) {
                tma::expect(a_bf16_sub_arrived[sub], quant_buf.bf16_tile);
                tma::load_async(
                    quant_buf.bf16_tile, g.A_bf16,
                    {0, sub},
                    a_bf16_sub_arrived[sub]);
            }
            wait(a_bf16_sub_arrived[sub], sub_phase);
            const uint32_t export_a_canonical_smem_base = static_cast<uint32_t>(
                __cvta_generic_to_shared(&a_export_fp4.data[0]));
            const uint32_t remote_a_scale_local_smem_base = static_cast<uint32_t>(
                __cvta_generic_to_shared(&a_recv_sc[0]));
            if constexpr (C::USE_CTA_AMAX) {
                quantize_operand_subtile_cta_amax(
                    quant_buf, local_tiles.A, local_tiles.sc, sub,
                    quant_sync_bar_id, &running_amax_a, warp_max_buf_a,
                    export_a_canonical_smem_base,
                    0,
                    0,
                    remote_a_scale_local_smem_base,
                    1,
                    cta_id == 0 && !C::BYPASS_SHARED_A_ON_CTA1);
            } else {
                quantize_operand_subtile_constant(
                    quant_buf, local_tiles.A, local_tiles.sc, sub,
                    export_a_canonical_smem_base,
                    0,
                    0,
                    remote_a_scale_local_smem_base,
                    1,
                    cta_id == 0 && !C::BYPASS_SHARED_A_ON_CTA1);
            }
            warpgroup::sync(quant_sync_bar_id);
        }
        sub_phase ^= 1;
        nvfp4_fused_gemm::copy_shared_local_bytes(
            static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(&local_tiles.sc.data[0])),
            sizeof(local_tiles.sc),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);

        if (cta_id == 0 && !C::BYPASS_SHARED_A_ON_CTA1) {
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            if (is_leader) {
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&a_recv_fp4),
                    reinterpret_cast<void *>(&a_export_fp4),
                    sizeof(typename G::a_export_t),
                    1,
                    a_payload_arrived);
            }
        }

        if (cta_id == 0) {
            if (g.debug_cta0_a_ptr != nullptr) {
                nvfp4_fused_gemm::dump_shared_raw_bytes(
                    &a_export_fp4.data[0], g.debug_cta0_a_ptr,
                    sizeof(a_export_fp4.data), local_tid, 128);
            }
            if (g.debug_cta0_sc_ptr != nullptr) {
                nvfp4_fused_gemm::dump_shared_raw_bytes(
                    &a_export_sc[0], g.debug_cta0_sc_ptr,
                    sizeof(a_export_sc), local_tid, 128);
            }
        } else {
            if (g.debug_cta1_a_ptr != nullptr) {
                if constexpr (C::DUMP_IMPORTED_TILE) {
                    nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                        local_tiles.A,
                        C::Mb,
                        C::Kb / 2,
                        g.debug_cta1_a_ptr,
                        g.debug_a_stride,
                        local_tid,
                        128);
                } else {
                    nvfp4_fused_gemm::dump_shared_raw_bytes(
                        &a_export_fp4.data[0], g.debug_cta1_a_ptr,
                        sizeof(a_export_fp4.data), local_tid, 128);
                }
            }
            if (g.debug_cta1_sc_ptr != nullptr) {
                if constexpr (C::DUMP_IMPORTED_TILE) {
                    nvfp4_fused_gemm::dump_shared_raw_bytes(
                        &local_tiles.sc.data[0], g.debug_cta1_sc_ptr,
                        sizeof(local_tiles.sc), local_tid, 128);
                } else {
                    nvfp4_fused_gemm::dump_shared_raw_bytes(
                        &a_export_sc[0], g.debug_cta1_sc_ptr,
                        sizeof(a_export_sc), local_tid, 128);
                }
            }
        }
    } else {
        wait(a_payload_arrived, 0);
        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if constexpr (C::DUMP_IMPORTED_TILE) {
            nvfp4_fused_gemm::copy_shared_local_bytes(
                static_cast<uint32_t>(__cvta_generic_to_shared(&local_tiles.sc.data[0])),
                static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc[0])),
                sizeof(local_tiles.sc),
                local_tid,
                128);
            warpgroup::sync(quant_sync_bar_id);
            nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                local_tiles.A,
                C::Mb,
                C::Kb / 2,
                static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4.data[0])),
                local_tid,
                128);
            warpgroup::sync(quant_sync_bar_id);
        }
        if (g.debug_cta1_a_ptr != nullptr) {
            if constexpr (C::DUMP_IMPORTED_TILE) {
                nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                    local_tiles.A,
                    C::Mb,
                    C::Kb / 2,
                    g.debug_cta1_a_ptr,
                    g.debug_a_stride,
                    local_tid,
                    128);
            } else {
                nvfp4_fused_gemm::dump_shared_raw_bytes(
                    &a_recv_fp4.data[0], g.debug_cta1_a_ptr,
                    sizeof(a_recv_fp4.data), local_tid, 128);
            }
        }
        if (g.debug_cta1_sc_ptr != nullptr) {
            if constexpr (C::DUMP_IMPORTED_TILE) {
                nvfp4_fused_gemm::dump_shared_raw_bytes(
                    &local_tiles.sc.data[0], g.debug_cta1_sc_ptr,
                    sizeof(local_tiles.sc), local_tid, 128);
            } else {
                nvfp4_fused_gemm::dump_shared_raw_bytes(
                    &a_recv_sc[0], g.debug_cta1_sc_ptr,
                    sizeof(a_recv_sc), local_tid, 128);
            }
        }
    }
}

template <typename C>
__device__ inline void clustered_transport_debug_kernel(const clustered_transport_debug_globals<C> &g) {
    using G = clustered_transport_debug_globals<C>;
    static_assert(C::CLUSTER_SIZE == 4, "clustered transport debug expects a 4-CTA cluster");
    static_assert(C::NUM_THREADS == 128, "clustered transport debug uses one warpgroup per CTA");

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
    }

    const int cta_id = cluster_ctarank();
    const int cta_row = cta_id / 2;
    const int cta_col = cta_id % 2;
    const bool own_a = (cta_col == 0);
    const bool own_b = (cta_row == 0);
    const int local_tid = threadIdx.x;
    const bool is_leader = (local_tid == 0);
    constexpr int quant_sync_bar_id = 0;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::local_tiles_t &local_tiles = sm_allocator.allocate<typename G::local_tiles_t>();
    typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();

    __shared__ typename G::export_t a_export_fp4;
    __shared__ typename G::scale_export_t a_export_sc;
    __shared__ typename G::export_t a_recv_fp4;
    __shared__ typename G::scale_export_t a_recv_sc;
    __shared__ typename G::export_t b_export_fp4;
    __shared__ typename G::scale_export_t b_export_sc;
    __shared__ typename G::export_t b_recv_fp4;
    __shared__ typename G::scale_export_t b_recv_sc;
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore b_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore a_payload_arrived;
    __shared__ semaphore a_scale_arrived;
    __shared__ semaphore b_payload_arrived;
    __shared__ semaphore b_scale_arrived;
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
            init_semaphore(b_bf16_sub_arrived[sub], 0, 1);
        }
        init_semaphore(a_payload_arrived, 0, 1);
        init_semaphore(a_scale_arrived, 0, 1);
        init_semaphore(b_payload_arrived, 0, 1);
        init_semaphore(b_scale_arrived, 0, 1);
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
        }
    }

    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait();
    __syncthreads();

    if constexpr (C::USE_CTA_AMAX) {
        if (own_a && is_leader) {
            running_amax_a = 0.0f;
        }
        if (own_b && is_leader) {
            running_amax_b = 0.0f;
        }
        warpgroup::sync(quant_sync_bar_id);
    }

    int sub_phase = 0;
    if (own_a) {
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            if (is_leader) {
                tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                tma::load_async(
                    quant_buf_a.bf16_tile,
                    g.A_bf16,
                    {cta_row, sub},
                    a_bf16_sub_arrived[sub]);
            }
            wait(a_bf16_sub_arrived[sub], sub_phase);
            const uint32_t export_a_canonical_smem_base =
                static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_fp4.data[0]));
            if constexpr (C::USE_CTA_AMAX) {
                quantize_operand_subtile_cta_amax(
                    quant_buf_a, local_tiles.A, local_tiles.A_sc, sub,
                    quant_sync_bar_id, &running_amax_a, warp_max_buf_a,
                    export_a_canonical_smem_base);
            } else {
                quantize_operand_subtile_constant(
                    quant_buf_a, local_tiles.A, local_tiles.A_sc, sub,
                    export_a_canonical_smem_base);
            }
            warpgroup::sync(quant_sync_bar_id);
        }
        sub_phase ^= 1;
        nvfp4_fused_gemm::copy_shared_local_bytes(
            static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc.data[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(&local_tiles.A_sc.data[0])),
            sizeof(local_tiles.A_sc),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);
    }

    sub_phase = 0;
    if (own_b) {
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            if (is_leader) {
                tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                tma::load_async(
                    quant_buf_b.bf16_tile,
                    g.B_bf16,
                    {cta_col, sub},
                    b_bf16_sub_arrived[sub]);
            }
            wait(b_bf16_sub_arrived[sub], sub_phase);
            const uint32_t export_b_canonical_smem_base =
                static_cast<uint32_t>(__cvta_generic_to_shared(&b_export_fp4.data[0]));
            if constexpr (C::USE_CTA_AMAX) {
                quantize_operand_subtile_cta_amax(
                    quant_buf_b, local_tiles.B, local_tiles.B_sc, sub,
                    quant_sync_bar_id, &running_amax_b, warp_max_buf_b,
                    export_b_canonical_smem_base);
            } else {
                quantize_operand_subtile_constant(
                    quant_buf_b, local_tiles.B, local_tiles.B_sc, sub,
                    export_b_canonical_smem_base);
            }
            warpgroup::sync(quant_sync_bar_id);
        }
        sub_phase ^= 1;
        nvfp4_fused_gemm::copy_shared_local_bytes(
            static_cast<uint32_t>(__cvta_generic_to_shared(&b_export_sc.data[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(&local_tiles.B_sc.data[0])),
            sizeof(local_tiles.B_sc),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);
    }

    __syncthreads();

    if (own_a && is_leader) {
        const int target_cta = cta_id + 1;
        tma::cluster::expect_bytes(a_payload_arrived, sizeof(typename G::export_t), target_cta);
        tma::cluster::expect_bytes(a_scale_arrived, sizeof(typename G::scale_export_t), target_cta);
        tma::cluster::store_async(
            reinterpret_cast<void *>(&a_recv_fp4),
            reinterpret_cast<void *>(&a_export_fp4),
            sizeof(typename G::export_t),
            target_cta,
            a_payload_arrived);
        tma::cluster::store_async(
            reinterpret_cast<void *>(&a_recv_sc),
            reinterpret_cast<void *>(&a_export_sc),
            sizeof(typename G::scale_export_t),
            target_cta,
            a_scale_arrived);
    }

    if (own_b && is_leader) {
        const int target_cta = cta_id + 2;
        tma::cluster::expect_bytes(b_payload_arrived, sizeof(typename G::export_t), target_cta);
        tma::cluster::expect_bytes(b_scale_arrived, sizeof(typename G::scale_export_t), target_cta);
        tma::cluster::store_async(
            reinterpret_cast<void *>(&b_recv_fp4),
            reinterpret_cast<void *>(&b_export_fp4),
            sizeof(typename G::export_t),
            target_cta,
            b_payload_arrived);
        tma::cluster::store_async(
            reinterpret_cast<void *>(&b_recv_sc),
            reinterpret_cast<void *>(&b_export_sc),
            sizeof(typename G::scale_export_t),
            target_cta,
            b_scale_arrived);
    }

    if (!own_a) {
        wait(a_payload_arrived, 0);
        wait(a_scale_arrived, 0);
        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        nvfp4_fused_gemm::copy_shared_local_bytes(
            static_cast<uint32_t>(__cvta_generic_to_shared(&local_tiles.A_sc.data[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc.data[0])),
            sizeof(local_tiles.A_sc),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);
        nvfp4_fused_gemm::import_local_canonical_fp4_tile(
            local_tiles.A,
            C::Mb,
            C::Kb / 2,
            static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4.data[0])),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);
    }

    if (!own_b) {
        wait(b_payload_arrived, 0);
        wait(b_scale_arrived, 0);
        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        nvfp4_fused_gemm::copy_shared_local_bytes(
            static_cast<uint32_t>(__cvta_generic_to_shared(&local_tiles.B_sc.data[0])),
            static_cast<uint32_t>(__cvta_generic_to_shared(&b_recv_sc.data[0])),
            sizeof(local_tiles.B_sc),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);
        nvfp4_fused_gemm::import_local_canonical_fp4_tile(
            local_tiles.B,
            C::Nb,
            C::Kb / 2,
            static_cast<uint32_t>(__cvta_generic_to_shared(&b_recv_fp4.data[0])),
            local_tid,
            128);
        warpgroup::sync(quant_sync_bar_id);
    }

    __syncthreads();

    if (cta_id == 0) {
        if (g.debug_a_owner_ptr != nullptr) {
            nvfp4_fused_gemm::dump_shared_raw_bytes(
                &a_export_fp4.data[0], g.debug_a_owner_ptr,
                sizeof(a_export_fp4.data), local_tid, 128);
        }
        if (g.debug_a_owner_sc_ptr != nullptr) {
            nvfp4_fused_gemm::dump_shared_raw_bytes(
                &a_export_sc.data[0], g.debug_a_owner_sc_ptr,
                sizeof(a_export_sc.data), local_tid, 128);
        }
        if (g.debug_b_owner_ptr != nullptr) {
            nvfp4_fused_gemm::dump_shared_raw_bytes(
                &b_export_fp4.data[0], g.debug_b_owner_ptr,
                sizeof(b_export_fp4.data), local_tid, 128);
        }
        if (g.debug_b_owner_sc_ptr != nullptr) {
            nvfp4_fused_gemm::dump_shared_raw_bytes(
                &b_export_sc.data[0], g.debug_b_owner_sc_ptr,
                sizeof(b_export_sc.data), local_tid, 128);
        }
    } else if (cta_id == 1) {
        if (g.debug_a_recv_ptr != nullptr) {
            nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                local_tiles.A,
                C::Mb,
                C::Kb / 2,
                g.debug_a_recv_ptr,
                g.debug_stride,
                local_tid,
                128);
        }
        if (g.debug_a_recv_sc_ptr != nullptr) {
            nvfp4_fused_gemm::dump_shared_raw_bytes(
                &local_tiles.A_sc.data[0], g.debug_a_recv_sc_ptr,
                sizeof(local_tiles.A_sc.data), local_tid, 128);
        }
    } else if (cta_id == 2) {
        if (g.debug_b_recv_ptr != nullptr) {
            nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                local_tiles.B,
                C::Nb,
                C::Kb / 2,
                g.debug_b_recv_ptr,
                g.debug_stride,
                local_tid,
                128);
        }
        if (g.debug_b_recv_sc_ptr != nullptr) {
            nvfp4_fused_gemm::dump_shared_raw_bytes(
                &local_tiles.B_sc.data[0], g.debug_b_recv_sc_ptr,
                sizeof(local_tiles.B_sc.data), local_tid, 128);
        }
    }
}

template <typename C>
__device__ inline void kernel_shared_a_cross_cta(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::SHARE_A_ACROSS_CTAS, "shared-A helper only supports shared-A configs");
    static_assert(C::CLUSTER_SIZE == 2 && C::COL_TILES_PER_BLOCK == 1,
                  "shared-A helper expects 2-CTA clusters and one local output tile per CTA");

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int row_block_idx = blockIdx.y;
    const int col_block_idx = clusterIdx().x;
    const int cta_col_idx = col_block_idx * C::CLUSTER_SIZE + cta_id;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
    typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::a_export_t (&a_recv_fp4_smem)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.template allocate<typename G::a_export_t, C::LOAD_PIPE_DEPTH>();
    typename G::a_scale_export_t (&a_recv_sc_smem)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.template allocate<typename G::a_scale_export_t, C::LOAD_PIPE_DEPTH>();
    tensor_allocator<1, C::TMEM_NCTA, false> tm_allocator;

    __shared__ typename G::a_export_t a_export_fp4[C::LOAD_PIPE_DEPTH];
    alignas(16) __shared__ uint8_t a_export_sc[C::LOAD_PIPE_DEPTH][sizeof(typename G::A_sc_tile)];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore b_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore a_stage_released[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_payload_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_payload_acked[C::LOAD_PIPE_DEPTH];
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            init_semaphore(a_quant_done[stage], 0, 1);
            init_semaphore(b_quant_done[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
            init_semaphore(a_stage_released[stage], 0, C::CLUSTER_SIZE);
            init_semaphore(a_payload_arrived[stage], 0, 1);
            init_semaphore(a_payload_acked[stage], 0, 1);
        }
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
            init_semaphore(b_bf16_sub_arrived[sub], 0, 1);
        }
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();
    const bool transport_only_debug = g.debug_transport_only;
    const bool main_dump_only_debug = g.debug_main_dump_only;
    constexpr bool remote_publish_active_top =
        !C::BYPASS_SHARED_A_ON_CTA1 || C::PUBLISH_SHARED_A_WHEN_BYPASSED;
    constexpr bool fire_and_forget_publish_debug =
        C::BYPASS_SHARED_A_ON_CTA1 && C::PUBLISH_SHARED_A_WHEN_BYPASSED;
    everyone::tma::cluster::wait();
    __syncthreads();

    if constexpr (remote_publish_active_top) {
        if (!transport_only_debug && !main_dump_only_debug) {
            // For the single-stage reproducer, pre-arm CTA1's receive semaphore before
            // any CTA0 publish can happen. The exact transport-only probe already does
            // this, and the live kernel was missing the equivalent receiver-ready sync.
            if (!fire_and_forget_publish_debug &&
                num_red_blocks == 1 &&
                cta_id == 1 &&
                warpgroup_id == 1 &&
                (threadIdx.x % 128) == 0) {
                tma::expect_bytes(a_payload_arrived[0], sizeof(typename G::a_export_t));
            }
            everyone::tma::cluster::sync();
        }
    }

    if (transport_only_debug || main_dump_only_debug) {
        constexpr int transport_fp4_bytes = C::Mb * (C::Kb / 2);
        const uint32_t transport_recv_fp4_smem_base =
            main_dump_only_debug
                ? static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4_smem[0].data[0]))
                : static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<uint8_t *>(&quant_buf_b)));
        const uint32_t transport_recv_sc_smem_base =
            main_dump_only_debug
                ? static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc_smem[0].data[0]))
                : (transport_recv_fp4_smem_base + transport_fp4_bytes);
        if constexpr (!C::BYPASS_SHARED_A_ON_CTA1) {
            everyone::tma::cluster::sync();
        }

        int sub_phase = 0;
        if constexpr (C::USE_CTA_AMAX) {
            if (warpgroup_id == 1 && (threadIdx.x % 128) == 0 && (cta_id == 0 || C::BYPASS_SHARED_A_ON_CTA1)) {
                running_amax_a = 0.0f;
            }
        }
        __syncthreads();

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            const int debug_stage = main_dump_only_debug ? (red_block_idx % C::LOAD_PIPE_DEPTH) : 0;
            const uint32_t debug_transport_recv_fp4_smem_base =
                main_dump_only_debug
                    ? static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4_smem[debug_stage].data[0]))
                    : transport_recv_fp4_smem_base;
            const uint32_t debug_transport_recv_sc_smem_base =
                main_dump_only_debug
                    ? static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc_smem[debug_stage].data[0]))
                    : transport_recv_sc_smem_base;
            if (warpgroup_id == 1) {
                const int local_tid = threadIdx.x % 128;
                const bool is_leader = (local_tid == 0);
                constexpr int a_quant_bar_id = 2;

                if (cta_id == 0) {
                    #pragma unroll
                    for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                        if (is_leader) {
                            tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                            tma::load_async(
                                quant_buf_a.bf16_tile, g.A_bf16,
                                {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                                a_bf16_sub_arrived[sub]);
                        }
                        wait(a_bf16_sub_arrived[sub], sub_phase);
                        const uint32_t export_a_canonical_smem_base = static_cast<uint32_t>(
                            __cvta_generic_to_shared(&a_export_fp4[debug_stage].data[0]));
                        const uint32_t remote_a_canonical_smem_base =
                            (!C::BYPASS_SHARED_A_ON_CTA1 && !main_dump_only_debug)
                                ? debug_transport_recv_fp4_smem_base
                                : 0;
                        const uint32_t remote_a_scale_local_smem_base =
                            (!C::BYPASS_SHARED_A_ON_CTA1)
                                ? debug_transport_recv_sc_smem_base
                                : 0;
                        if constexpr (C::USE_CTA_AMAX) {
                            quantize_operand_subtile_cta_amax(
                                quant_buf_a, input_tiles[0].A, input_scales[0].A, sub,
                                a_quant_bar_id, &running_amax_a, warp_max_buf_a,
                                export_a_canonical_smem_base,
                                0,
                                remote_a_canonical_smem_base,
                                remote_a_scale_local_smem_base,
                                1,
                                !C::BYPASS_SHARED_A_ON_CTA1);
                        } else {
                            quantize_operand_subtile_constant(
                                quant_buf_a, input_tiles[0].A, input_scales[0].A, sub,
                                export_a_canonical_smem_base,
                                0,
                                remote_a_canonical_smem_base,
                                remote_a_scale_local_smem_base,
                                1,
                                !C::BYPASS_SHARED_A_ON_CTA1);
                        }
                        warpgroup::sync(a_quant_bar_id);
                    }
                    nvfp4_fused_gemm::copy_shared_local_bytes(
                        static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc[debug_stage][0])),
                        static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[0].A.data[0])),
                        sizeof(input_scales[0].A),
                        local_tid,
                        128);
                    warpgroup::sync(a_quant_bar_id);
                    if constexpr (!C::BYPASS_SHARED_A_ON_CTA1) {
                        if (main_dump_only_debug) {
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            nvfp4_fused_gemm::mirror_shared_bytes_to_cluster(
                                static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_fp4[debug_stage].data[0])),
                                debug_transport_recv_fp4_smem_base,
                                1,
                                sizeof(a_export_fp4[debug_stage].data),
                                local_tid,
                                128);
                            warpgroup::sync(a_quant_bar_id);
                        }
                    }

                    if (g.debug_cta0_a_ptr != nullptr) {
                        if constexpr (C::BYPASS_SHARED_A_ON_CTA1) {
                            nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                                input_tiles[0].A,
                                C::Mb,
                                C::Kb / 2,
                                g.debug_cta0_a_ptr,
                                g.debug_a_stride,
                                local_tid,
                                128);
                        } else {
                            nvfp4_fused_gemm::dump_shared_raw_bytes(
                                &a_export_fp4[debug_stage].data[0], g.debug_cta0_a_ptr,
                                C::Mb * (C::Kb / 2), local_tid, 128);
                        }
                    }
                    if (g.debug_cta0_sc_ptr != nullptr) {
                        if constexpr (C::BYPASS_SHARED_A_ON_CTA1) {
                            nvfp4_fused_gemm::dump_shared_raw_bytes(
                                &input_scales[0].A.data[0], g.debug_cta0_sc_ptr,
                                sizeof(input_scales[0].A), local_tid, 128);
                        } else {
                            nvfp4_fused_gemm::dump_shared_raw_bytes(
                                &a_export_sc[debug_stage][0], g.debug_cta0_sc_ptr,
                                sizeof(input_scales[0].A), local_tid, 128);
                        }
                    }
                } else if constexpr (C::BYPASS_SHARED_A_ON_CTA1) {
                    #pragma unroll
                    for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                        if (is_leader) {
                            tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                            tma::load_async(
                                quant_buf_a.bf16_tile, g.A_bf16,
                                {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                                a_bf16_sub_arrived[sub]);
                        }
                        wait(a_bf16_sub_arrived[sub], sub_phase);
                        if constexpr (C::USE_CTA_AMAX) {
                            quantize_operand_subtile_cta_amax(
                                quant_buf_a, input_tiles[0].A, input_scales[0].A, sub,
                                a_quant_bar_id, &running_amax_a, warp_max_buf_a);
                        } else {
                            quantize_operand_subtile_constant(
                                quant_buf_a, input_tiles[0].A, input_scales[0].A, sub);
                        }
                        warpgroup::sync(a_quant_bar_id);
                    }
                    warpgroup::sync(a_quant_bar_id);
                }
            }

            __syncthreads();

            if constexpr (!C::BYPASS_SHARED_A_ON_CTA1) {
                if (cta_id == 0 && threadIdx.x == 0) {
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    nvfp4_fused_gemm::arrive_remote_cluster(a_payload_arrived[debug_stage], 1, 1);
                }

                if (cta_id == 1 && warpgroup_id == 1) {
                    wait(a_payload_arrived[debug_stage], 0);
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                }
                __syncthreads();

                if (cta_id == 1 && warpgroup_id == 1) {
                    const int local_tid = threadIdx.x % 128;
                    nvfp4_fused_gemm::copy_shared_local_bytes(
                        static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[0].A.data[0])),
                        debug_transport_recv_sc_smem_base,
                        sizeof(input_scales[0].A),
                        local_tid,
                        128);
                    warpgroup::sync(2);
                    nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                        input_tiles[0].A,
                        C::Mb,
                        C::Kb / 2,
                        debug_transport_recv_fp4_smem_base,
                        local_tid,
                        128);
                    warpgroup::sync(2);
                    if (g.debug_cta1_a_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                            input_tiles[0].A,
                            C::Mb,
                            C::Kb / 2,
                            g.debug_cta1_a_ptr,
                            g.debug_a_stride,
                            local_tid,
                            128);
                    }
                    if (g.debug_cta1_sc_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_shared_raw_bytes(
                            &input_scales[0].A.data[0], g.debug_cta1_sc_ptr,
                            sizeof(input_scales[0].A), local_tid, 128);
                    }
                }
            } else if (cta_id == 1 && warpgroup_id == 1) {
                const int local_tid = threadIdx.x % 128;
                if (g.debug_cta1_a_ptr != nullptr) {
                    nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                        input_tiles[0].A,
                        C::Mb,
                        C::Kb / 2,
                        g.debug_cta1_a_ptr,
                        g.debug_a_stride,
                        local_tid,
                        128);
                }
                if (g.debug_cta1_sc_ptr != nullptr) {
                    nvfp4_fused_gemm::dump_shared_raw_bytes(
                        &input_scales[0].A.data[0], g.debug_cta1_sc_ptr,
                        sizeof(input_scales[0].A), local_tid, 128);
                }
            }

            __syncthreads();
            sub_phase ^= 1;
        }
        __syncthreads();
        return;
    }

    if constexpr (!fire_and_forget_publish_debug) {
        if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        if (warpgroup_id == 0 || warpgroup_id == 3) {
            tm_allocator.set_addr(tmem_addr);
        }
    }

    if (warpgroup_id == 1) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        constexpr bool remote_publish_active =
            !C::BYPASS_SHARED_A_ON_CTA1 || C::PUBLISH_SHARED_A_WHEN_BYPASSED;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        uint32_t release_phasebits = 0xFFFF0000;
        uint32_t payload_arrived_phasebits = 0xFFFF0000;
        uint32_t payload_acked_phasebits = 0xFFFF0000;
        int sub_phase = 0;
        int iter_idx = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader && (cta_id == 0 || C::BYPASS_SHARED_A_ON_CTA1)) {
                running_amax_a = 0.0f;
            }
            warpgroup::sync(a_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx, ++iter_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            if constexpr (remote_publish_active) {
                if (!fire_and_forget_publish_debug &&
                    cta_id == 1 &&
                    is_leader &&
                    num_red_blocks != 1) {
                    tma::expect_bytes(a_payload_arrived[stage], sizeof(typename G::a_export_t));
                }
            }

            if (cta_id == 0) {
                if (iter_idx >= C::LOAD_PIPE_DEPTH) {
                    if (is_leader) {
                        arrive(a_stage_released[stage], 1);
                    }
                    wait(a_stage_released[stage], get_phasebit<1>(release_phasebits, stage));
                }

                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                        tma::load_async(
                            quant_buf_a.bf16_tile, g.A_bf16,
                            {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            a_bf16_sub_arrived[sub]);
                    }
                    wait(a_bf16_sub_arrived[sub], sub_phase);
                    const uint32_t export_a_canonical_smem_base = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&a_export_fp4[stage].data[0]));
                    const uint32_t remote_a_canonical_smem_base = 0;
                    const uint32_t remote_a_scale_local_smem_base =
                        (!C::BYPASS_SHARED_A_ON_CTA1)
                            ? static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc_smem[stage].data[0]))
                            : 0;
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            a_quant_bar_id, &running_amax_a, warp_max_buf_a,
                            export_a_canonical_smem_base,
                            0,
                            remote_a_canonical_smem_base,
                            remote_a_scale_local_smem_base,
                            1,
                            remote_publish_active);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            export_a_canonical_smem_base,
                            0,
                            remote_a_canonical_smem_base,
                            remote_a_scale_local_smem_base,
                            1,
                            remote_publish_active);
                    }
                    warpgroup::sync(a_quant_bar_id);
                }
                sub_phase ^= 1;
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc[stage][0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                    sizeof(input_scales[stage].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                if constexpr (remote_publish_active) {
                    const uint32_t remote_payload_local_smem_base =
                        static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                (C::BYPASS_SHARED_A_ON_CTA1 && C::PUBLISH_SHARED_A_WHEN_BYPASSED)
                                    ? &a_export_fp4[stage].data[0]
                                    : &a_recv_fp4_smem[stage].data[0]));
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    if constexpr (fire_and_forget_publish_debug) {
                        if (is_leader) {
                            nvfp4_fused_gemm::mirror_shared_bytes_to_cluster(
                                static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_fp4[stage].data[0])),
                                remote_payload_local_smem_base,
                                1,
                                sizeof(typename G::a_export_t),
                                0,
                                1);
                        }
                    } else {
                        nvfp4_fused_gemm::mirror_shared_bytes_to_cluster(
                            static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_fp4[stage].data[0])),
                            remote_payload_local_smem_base,
                            1,
                            sizeof(typename G::a_export_t),
                            local_tid,
                            128);
                    }
                    warpgroup::sync(a_quant_bar_id);
                    if (is_leader && !fire_and_forget_publish_debug) {
                        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                        nvfp4_fused_gemm::arrive_remote_cluster(a_payload_arrived[stage], 1, 1);
                    }
                }

                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    if constexpr (remote_publish_active && !fire_and_forget_publish_debug) {
                        wait(a_payload_acked[stage], get_phasebit<1>(payload_acked_phasebits, stage));
                        update_phasebit<1>(payload_acked_phasebits, stage);
                    }
                    arrive(a_quant_done[stage], 1);
                }
                if (row_block_idx == 0 && red_block_idx == 0) {
                    if (g.debug_cta0_a_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_shared_raw_bytes(
                            &a_export_fp4[stage].data[0], g.debug_cta0_a_ptr,
                            C::Mb * (C::Kb / 2), local_tid, 128);
                    }
                    if (g.debug_cta0_sc_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_shared_raw_bytes(
                            &a_export_sc[stage][0], g.debug_cta0_sc_ptr,
                            sizeof(input_scales[stage].A), local_tid, 128);
                    }
                }
                if (iter_idx >= C::LOAD_PIPE_DEPTH) {
                    update_phasebit<1>(release_phasebits, stage);
                }
            } else if constexpr (C::BYPASS_SHARED_A_ON_CTA1) {
                if (iter_idx >= C::LOAD_PIPE_DEPTH && is_leader) {
                    nvfp4_fused_gemm::arrive_remote_cluster(a_stage_released[stage], 0, 1);
                }

                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                        tma::load_async(
                            quant_buf_a.bf16_tile, g.A_bf16,
                            {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            a_bf16_sub_arrived[sub]);
                    }
                    wait(a_bf16_sub_arrived[sub], sub_phase);
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            a_quant_bar_id, &running_amax_a, warp_max_buf_a);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub);
                    }
                    warpgroup::sync(a_quant_bar_id);
                }
                sub_phase ^= 1;
                if (row_block_idx == 0 && red_block_idx == 0) {
                    if (g.debug_cta1_a_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                            input_tiles[stage].A,
                            C::Mb,
                            C::Kb / 2,
                            g.debug_cta1_a_ptr,
                            g.debug_a_stride,
                            local_tid,
                            128);
                    }
                    if (g.debug_cta1_sc_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_shared_raw_bytes(
                            &input_scales[stage].A.data[0], g.debug_cta1_sc_ptr,
                            sizeof(input_scales[stage].A), local_tid, 128);
                    }
                }
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    if constexpr (remote_publish_active && !fire_and_forget_publish_debug) {
                        wait(a_payload_arrived[stage], get_phasebit<1>(payload_arrived_phasebits, stage));
                        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        nvfp4_fused_gemm::arrive_remote_cluster(a_payload_acked[stage], 0, 1);
                        update_phasebit<1>(payload_arrived_phasebits, stage);
                    }
                    arrive(a_quant_done[stage], 1);
                }
            } else {
                if (iter_idx >= C::LOAD_PIPE_DEPTH && is_leader) {
                    __threadfence_block();
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    nvfp4_fused_gemm::arrive_remote_cluster(a_stage_released[stage], 0, 1);
                }
                wait(a_payload_arrived[stage], get_phasebit<1>(payload_arrived_phasebits, stage));
                warpgroup::sync(a_quant_bar_id);
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc_smem[stage].data[0])),
                    sizeof(input_scales[stage].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                    input_tiles[stage].A,
                    C::Mb,
                    C::Kb / 2,
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4_smem[stage].data[0])),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                if (row_block_idx == 0 && red_block_idx == 0) {
                    if (g.debug_cta1_a_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_swizzled_fp4_tile_canonical(
                            input_tiles[stage].A,
                            C::Mb,
                            C::Kb / 2,
                            g.debug_cta1_a_ptr,
                            g.debug_a_stride,
                            local_tid,
                            128);
                    }
                    if (g.debug_cta1_sc_ptr != nullptr) {
                        nvfp4_fused_gemm::dump_shared_raw_bytes(
                            &input_scales[stage].A.data[0], g.debug_cta1_sc_ptr,
                            sizeof(input_scales[stage].A), local_tid, 128);
                    }
                }
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    nvfp4_fused_gemm::arrive_remote_cluster(a_payload_acked[stage], 0, 1);
                    arrive(a_quant_done[stage], 1);
                }
                update_phasebit<1>(payload_arrived_phasebits, stage);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if constexpr (!fire_and_forget_publish_debug) if (warpgroup_id == 2) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int b_quant_bar_id = 3;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_b = 0.0f;
            }
            warpgroup::sync(b_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            if constexpr (C::PUBLISH_SHARED_A_WHEN_BYPASSED) {
                // Debug-only: serialize the B-side pipeline behind the shared-A publish
                // handshake so we can tell whether the remaining launch failure is caused
                // by sender-side remote payload traffic overlapping the live B pipeline.
                wait(a_quant_done[stage], get_phasebit<1>(phasebits, stage));
            }
            #pragma unroll
            for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                if (is_leader) {
                    tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                    tma::load_async(
                        quant_buf_b.bf16_tile, g.B_bf16,
                        {cta_col_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                        b_bf16_sub_arrived[sub]);
                }
                wait(b_bf16_sub_arrived[sub], sub_phase);
                if constexpr (C::USE_CTA_AMAX) {
                    quantize_operand_subtile_cta_amax(
                        quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub,
                        b_quant_bar_id, &running_amax_b, warp_max_buf_b);
                } else {
                    quantize_operand_subtile_constant(
                        quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub);
                }
                warpgroup::sync(b_quant_bar_id);
            }
            sub_phase ^= 1;

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (is_leader) {
                arrive(b_quant_done[stage], 1);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if constexpr (!fire_and_forget_publish_debug) if (warpgroup_id == 3 && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 0) {
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            uint32_t stage = 0;
            uint32_t mma_phasebits = 0;

            tensor_after_thread_sync();
            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                #pragma unroll
                for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                    auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &A_sc_sm_subtile =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                    auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &B_sc_sm_subtile =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                }

                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                if (red_block_idx == 0) {
                    mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                } else {
                    mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                }
                kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                mma_phasebits ^= (1u << stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
            tensor_commit<1>(outputs_arrived);
        }
    }

    if constexpr (!fire_and_forget_publish_debug) if (warpgroup_id == 0) {
        wait(outputs_arrived, 0);
        const float a_sg_dec = C::USE_CTA_AMAX
            ? (((cta_id == 0 || C::BYPASS_SHARED_A_ON_CTA1)
                    ? running_amax_a
                    : nvfp4_fused_gemm::load_running_amax_remote<C>(&running_amax_a, 0)) /
               (6.0f * 448.0f))
            : SCALE_MAX_DEC;
        const float b_sg_dec = C::USE_CTA_AMAX ? running_amax_b / (6.0f * 448.0f) : SCALE_MAX_DEC;
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
            warpgroup::load_async(
                D_reg_fl,
                out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * epi));
            warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
            warp::copy(D_reg[epi], D_reg_fl);
        }

        tensor_load_wait();
        tensor_before_thread_sync();
        warpgroup::sync(1);

        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                g.D, output_tiles.D[epi % C::NUM_D_TILES],
                {row_block_idx, C::EPI_PIPE_DEPTH * cta_col_idx + epi});
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
}

template <typename C>
__device__ inline void kernel_cluster_reuse_2d(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::CLUSTER_REUSE_2D, "clustered 2D helper only supports clustered configs");
    static_assert(C::CLUSTER_SIZE == 4 && C::COL_TILES_PER_BLOCK == 1,
                  "clustered 2D helper expects 4-CTA clusters with one output tile per CTA");

    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cta_row = cta_id / 2;
    const int cta_col = cta_id % 2;
    const bool own_a = (cta_col == 0);
    const bool own_b = (cta_row == 0);
    const int cluster_linear = clusterIdx().x;
    const int clusters_per_row = g.D.cols() / (2 * C::Nb);
    const int cluster_row = cluster_linear / clusters_per_row;
    const int cluster_col = cluster_linear % clusters_per_row;
    const int row_block_idx = cluster_row * 2 + cta_row;
    const int col_block_idx = cluster_col * 2 + cta_col;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;
    const bool transport_only = g.debug_transport_only;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
    typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();
    tensor_allocator<1, 1, false> tm_allocator;

    __shared__ typename G::a_export_t a_export_fp4[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::a_scale_export_t a_export_sc[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::a_export_t a_recv_fp4[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::a_scale_export_t a_recv_sc[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::b_export_t b_export_fp4[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::b_scale_export_t b_export_sc[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::b_export_t b_recv_fp4[C::LOAD_PIPE_DEPTH];
    __shared__ typename G::b_scale_export_t b_recv_sc[C::LOAD_PIPE_DEPTH];
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore b_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore a_payload_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_scale_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_payload_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_scale_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_amax_arrived;
    __shared__ semaphore b_amax_arrived;
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float remote_amax_a;
    __shared__ float remote_amax_b;
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            init_semaphore(a_quant_done[stage], 0, 1);
            init_semaphore(b_quant_done[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
            init_semaphore(a_payload_arrived[stage], 0, 1);
            init_semaphore(a_scale_arrived[stage], 0, 1);
            init_semaphore(b_payload_arrived[stage], 0, 1);
            init_semaphore(b_scale_arrived[stage], 0, 1);
        }
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
            init_semaphore(b_bf16_sub_arrived[sub], 0, 1);
        }
        init_semaphore(a_amax_arrived, 0, 1);
        init_semaphore(b_amax_arrived, 0, 1);
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
            remote_amax_a = 0.0f;
            remote_amax_b = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (!transport_only) {
        if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        if (warpgroup_id == 0 || warpgroup_id == 3) {
            tm_allocator.set_addr(tmem_addr);
        }
    }
    everyone::tma::cluster::wait();
    __syncthreads();

    if (transport_only) {
        static_assert(C::LOAD_PIPE_DEPTH == 1, "clustered transport-only bring-up expects one stage");
        const int local_tid = threadIdx.x % 128;
        const bool is_wg1 = (warpgroup_id == 1);
        const bool is_wg2 = (warpgroup_id == 2);
        const bool is_wg_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        constexpr int b_quant_bar_id = 3;
        int a_sub_phase = 0;
        int b_sub_phase = 0;
        int transport_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_wg1 && own_a && is_wg_leader) {
                running_amax_a = 0.0f;
            }
            if (is_wg2 && own_b && is_wg_leader) {
                running_amax_b = 0.0f;
            }
            if (is_wg1) {
                warpgroup::sync(a_quant_bar_id);
            }
            if (is_wg2) {
                warpgroup::sync(b_quant_bar_id);
            }
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            if (is_wg1 && own_a) {
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_wg_leader) {
                        tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                        tma::load_async(
                            quant_buf_a.bf16_tile, g.A_bf16,
                            {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            a_bf16_sub_arrived[sub]);
                    }
                    wait(a_bf16_sub_arrived[sub], a_sub_phase);
                    const uint32_t export_a_canonical_smem_base =
                        static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_fp4[0].data[0]));
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_a, input_tiles[0].A, input_scales[0].A, sub,
                            a_quant_bar_id, &running_amax_a, warp_max_buf_a,
                            export_a_canonical_smem_base);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_a, input_tiles[0].A, input_scales[0].A, sub,
                            export_a_canonical_smem_base);
                    }
                    warpgroup::sync(a_quant_bar_id);
                }
                a_sub_phase ^= 1;
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc[0].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[0].A.data[0])),
                    sizeof(input_scales[0].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            }

            __syncthreads();

            if (threadIdx.x == 0 && own_a) {
                const int target_cta = cta_id + 1;
                tma::cluster::expect_bytes(a_payload_arrived[0], sizeof(typename G::a_export_t), target_cta);
                tma::cluster::expect_bytes(a_scale_arrived[0], sizeof(typename G::a_scale_export_t), target_cta);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&a_recv_fp4[0]),
                    reinterpret_cast<void *>(&a_export_fp4[0]),
                    sizeof(typename G::a_export_t),
                    target_cta,
                    a_payload_arrived[0]);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&a_recv_sc[0]),
                    reinterpret_cast<void *>(&a_export_sc[0]),
                    sizeof(typename G::a_scale_export_t),
                    target_cta,
                    a_scale_arrived[0]);
            }

            __syncthreads();

            if (is_wg1 && !own_a) {
                wait(a_payload_arrived[0], transport_phase);
                wait(a_scale_arrived[0], transport_phase);
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[0].A.data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc[0].data[0])),
                    sizeof(input_scales[0].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                    input_tiles[0].A,
                    C::Mb,
                    C::Kb / 2,
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4[0].data[0])),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            }

            __syncthreads();

            if (is_wg2 && own_b) {
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_wg_leader) {
                        tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                        tma::load_async(
                            quant_buf_b.bf16_tile, g.B_bf16,
                            {col_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            b_bf16_sub_arrived[sub]);
                    }
                    wait(b_bf16_sub_arrived[sub], b_sub_phase);
                    const uint32_t export_b_canonical_smem_base =
                        static_cast<uint32_t>(__cvta_generic_to_shared(&b_export_fp4[0].data[0]));
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_b, input_tiles[0].B[0], input_scales[0].B[0], sub,
                            b_quant_bar_id, &running_amax_b, warp_max_buf_b,
                            export_b_canonical_smem_base);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_b, input_tiles[0].B[0], input_scales[0].B[0], sub,
                            export_b_canonical_smem_base);
                    }
                    warpgroup::sync(b_quant_bar_id);
                }
                b_sub_phase ^= 1;
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&b_export_sc[0].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[0].B[0].data[0])),
                    sizeof(input_scales[0].B[0]),
                    local_tid,
                    128);
                warpgroup::sync(b_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            }

            __syncthreads();

            if (threadIdx.x == 0 && own_b) {
                const int target_cta = cta_id + 2;
                tma::cluster::expect_bytes(b_payload_arrived[0], sizeof(typename G::b_export_t), target_cta);
                tma::cluster::expect_bytes(b_scale_arrived[0], sizeof(typename G::b_scale_export_t), target_cta);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&b_recv_fp4[0]),
                    reinterpret_cast<void *>(&b_export_fp4[0]),
                    sizeof(typename G::b_export_t),
                    target_cta,
                    b_payload_arrived[0]);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&b_recv_sc[0]),
                    reinterpret_cast<void *>(&b_export_sc[0]),
                    sizeof(typename G::b_scale_export_t),
                    target_cta,
                    b_scale_arrived[0]);
            }

            __syncthreads();

            if (is_wg2 && !own_b) {
                wait(b_payload_arrived[0], transport_phase);
                wait(b_scale_arrived[0], transport_phase);
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[0].B[0].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&b_recv_sc[0].data[0])),
                    sizeof(input_scales[0].B[0]),
                    local_tid,
                    128);
                warpgroup::sync(b_quant_bar_id);
                nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                    input_tiles[0].B[0],
                    C::Nb,
                    C::Kb / 2,
                    static_cast<uint32_t>(__cvta_generic_to_shared(&b_recv_fp4[0].data[0])),
                    local_tid,
                    128);
                warpgroup::sync(b_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            }

            __syncthreads();
            transport_phase ^= 1;
        }
        return;
    }

    if (warpgroup_id == 1) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (own_a && is_leader) {
                running_amax_a = 0.0f;
            }
            warpgroup::sync(a_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            if (own_a) {
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                        tma::load_async(
                            quant_buf_a.bf16_tile, g.A_bf16,
                            {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            a_bf16_sub_arrived[sub]);
                    }
                    wait(a_bf16_sub_arrived[sub], sub_phase);
                    const uint32_t export_a_canonical_smem_base =
                        static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_fp4[stage].data[0]));
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            a_quant_bar_id, &running_amax_a, warp_max_buf_a,
                            export_a_canonical_smem_base);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            export_a_canonical_smem_base);
                    }
                    warpgroup::sync(a_quant_bar_id);
                }
                sub_phase ^= 1;
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_export_sc[stage].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                    sizeof(input_scales[stage].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (is_leader) {
                    const int target_cta = cta_id + 1;
                    tma::cluster::expect_bytes(a_payload_arrived[stage], sizeof(typename G::a_export_t), target_cta);
                    tma::cluster::expect_bytes(a_scale_arrived[stage], sizeof(typename G::a_scale_export_t), target_cta);
                    tma::cluster::store_async(
                        reinterpret_cast<void *>(&a_recv_fp4[stage]),
                        reinterpret_cast<void *>(&a_export_fp4[stage]),
                        sizeof(typename G::a_export_t),
                        target_cta,
                        a_payload_arrived[stage]);
                    tma::cluster::store_async(
                        reinterpret_cast<void *>(&a_recv_sc[stage]),
                        reinterpret_cast<void *>(&a_export_sc[stage]),
                        sizeof(typename G::a_scale_export_t),
                        target_cta,
                        a_scale_arrived[stage]);
                    arrive(a_quant_done[stage], 1);
                }
            } else {
                wait(a_payload_arrived[stage], get_phasebit<1>(phasebits, stage));
                wait(a_scale_arrived[stage], get_phasebit<1>(phasebits, stage));
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].A.data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_sc[stage].data[0])),
                    sizeof(input_scales[stage].A),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                    input_tiles[stage].A,
                    C::Mb,
                    C::Kb / 2,
                    static_cast<uint32_t>(__cvta_generic_to_shared(&a_recv_fp4[stage].data[0])),
                    local_tid,
                    128);
                warpgroup::sync(a_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    arrive(a_quant_done[stage], 1);
                }
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }

        if constexpr (C::USE_CTA_AMAX) {
            if (own_a && is_leader) {
                const int target_cta = cta_id + 1;
                tma::cluster::expect_bytes(a_amax_arrived, sizeof(float), target_cta);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&remote_amax_a),
                    reinterpret_cast<void *>(&running_amax_a),
                    sizeof(float),
                    target_cta,
                    a_amax_arrived);
            }
        }
    } else if (warpgroup_id == 2) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int b_quant_bar_id = 3;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (own_b && is_leader) {
                running_amax_b = 0.0f;
            }
            warpgroup::sync(b_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            if (transport_only) {
                wait(a_quant_done[stage], get_phasebit<1>(phasebits, stage));
            }
            if (own_b) {
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                        tma::load_async(
                            quant_buf_b.bf16_tile, g.B_bf16,
                            {col_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            b_bf16_sub_arrived[sub]);
                    }
                    wait(b_bf16_sub_arrived[sub], sub_phase);
                    const uint32_t export_b_canonical_smem_base =
                        static_cast<uint32_t>(__cvta_generic_to_shared(&b_export_fp4[stage].data[0]));
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub,
                            b_quant_bar_id, &running_amax_b, warp_max_buf_b,
                            export_b_canonical_smem_base);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub,
                            export_b_canonical_smem_base);
                    }
                    warpgroup::sync(b_quant_bar_id);
                }
                sub_phase ^= 1;
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&b_export_sc[stage].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].B[0].data[0])),
                    sizeof(input_scales[stage].B[0]),
                    local_tid,
                    128);
                warpgroup::sync(b_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (is_leader) {
                    const int target_cta = cta_id + 2;
                    tma::cluster::expect_bytes(b_payload_arrived[stage], sizeof(typename G::b_export_t), target_cta);
                    tma::cluster::expect_bytes(b_scale_arrived[stage], sizeof(typename G::b_scale_export_t), target_cta);
                    tma::cluster::store_async(
                        reinterpret_cast<void *>(&b_recv_fp4[stage]),
                        reinterpret_cast<void *>(&b_export_fp4[stage]),
                        sizeof(typename G::b_export_t),
                        target_cta,
                        b_payload_arrived[stage]);
                    tma::cluster::store_async(
                        reinterpret_cast<void *>(&b_recv_sc[stage]),
                        reinterpret_cast<void *>(&b_export_sc[stage]),
                        sizeof(typename G::b_scale_export_t),
                        target_cta,
                        b_scale_arrived[stage]);
                    arrive(b_quant_done[stage], 1);
                }
            } else {
                wait(b_payload_arrived[stage], get_phasebit<1>(phasebits, stage));
                wait(b_scale_arrived[stage], get_phasebit<1>(phasebits, stage));
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                nvfp4_fused_gemm::copy_shared_local_bytes(
                    static_cast<uint32_t>(__cvta_generic_to_shared(&input_scales[stage].B[0].data[0])),
                    static_cast<uint32_t>(__cvta_generic_to_shared(&b_recv_sc[stage].data[0])),
                    sizeof(input_scales[stage].B[0]),
                    local_tid,
                    128);
                warpgroup::sync(b_quant_bar_id);
                nvfp4_fused_gemm::import_local_canonical_fp4_tile(
                    input_tiles[stage].B[0],
                    C::Nb,
                    C::Kb / 2,
                    static_cast<uint32_t>(__cvta_generic_to_shared(&b_recv_fp4[stage].data[0])),
                    local_tid,
                    128);
                warpgroup::sync(b_quant_bar_id);
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    arrive(b_quant_done[stage], 1);
                }
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }

        if constexpr (C::USE_CTA_AMAX) {
            if (own_b && is_leader) {
                const int target_cta = cta_id + 2;
                tma::cluster::expect_bytes(b_amax_arrived, sizeof(float), target_cta);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&remote_amax_b),
                    reinterpret_cast<void *>(&running_amax_b),
                    sizeof(float),
                    target_cta,
                    b_amax_arrived);
            }
        }

        if (transport_only) {
            return;
        }
    } else if (warpgroup_id == 3 && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 0) {
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            uint32_t stage = 0;
            uint32_t mma_phasebits = 0;

            tensor_after_thread_sync();
            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                #pragma unroll
                for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                    auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &A_sc_sm_subtile =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                    auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                    auto &B_sc_sm_subtile =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                    load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                }

                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                if (red_block_idx == 0) {
                    mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                } else {
                    mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                }
                kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                mma_phasebits ^= (1u << stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
            tensor_commit<1>(outputs_arrived);
        }
    }

    if (warpgroup_id == 0) {
        wait(outputs_arrived, 0);
        float a_amax = 0.0f;
        float b_amax = 0.0f;
        if constexpr (C::USE_CTA_AMAX) {
            if (own_a) {
                a_amax = running_amax_a;
            } else {
                wait(a_amax_arrived, 0);
                a_amax = remote_amax_a;
            }
            if (own_b) {
                b_amax = running_amax_b;
            } else {
                wait(b_amax_arrived, 0);
                b_amax = remote_amax_b;
            }
        }
        const float a_sg_dec = C::USE_CTA_AMAX ? a_amax / (6.0f * 448.0f) : SCALE_MAX_DEC;
        const float b_sg_dec = C::USE_CTA_AMAX ? b_amax / (6.0f * 448.0f) : SCALE_MAX_DEC;
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
            warpgroup::load_async(
                D_reg_fl,
                out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                    0, C::Nb / C::EPI_PIPE_DEPTH * epi));
            warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
            warp::copy(D_reg[epi], D_reg_fl);
        }

        tensor_load_wait();
        tensor_before_thread_sync();
        warpgroup::sync(1);

        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                g.D, output_tiles.D[epi % C::NUM_D_TILES],
                {row_block_idx, C::EPI_PIPE_DEPTH * col_block_idx + epi});
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
}

template <typename C>
__device__ inline void kernel_shared_b_cross_cta(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::SHARE_B_ACROSS_CTAS, "shared-B helper only supports shared-B configs");
    static_assert(C::CLUSTER_SIZE == 2, "shared-B helper expects two-CTA clusters");
    static_assert(C::TMEM_NCTA == 2, "shared-B helper expects pooled 2-CTA TMEM");
    static_assert(C::DEDICATED_PRODUCER, "shared-B helper expects dedicated-producer configs");
    static_assert(C::COL_TILES_PER_BLOCK == 1, "shared-B helper expects one column tile per CTA");
    static_assert(C::Kb == 128, "shared-B helper currently supports Kb=128 only");
    static_assert(C::QUANT_SUB_TILES == 1, "shared-B helper currently expects one quant subtile per stage");
    const bool transport_only = g.debug_transport_only;
    const bool producer_only = transport_only && g.debug_front_half_mode == 1;
    const bool a_only = transport_only && g.debug_front_half_mode == 2;
    const bool b_only = transport_only && g.debug_front_half_mode == 3;
    const bool a_wait_only = transport_only && g.debug_front_half_mode == 4;
    const bool b_wait_only = transport_only && g.debug_front_half_mode == 5;
    const bool a_quant_only = transport_only && g.debug_front_half_mode == 6;
    const bool b_quant_only = transport_only && g.debug_front_half_mode == 7;
    const bool a_stage1_quant_only = transport_only && g.debug_front_half_mode == 8;
    const bool b_stage1_quant_only = transport_only && g.debug_front_half_mode == 9;
    const bool a_quant_per_stage_only = transport_only && g.debug_front_half_mode == 10;
    const bool b_quant_per_stage_only = transport_only && g.debug_front_half_mode == 11;
    const bool a_quant_then_wait_only = transport_only && g.debug_front_half_mode == 12;
    const bool b_quant_then_wait_only = transport_only && g.debug_front_half_mode == 13;
    const bool a_quant_then_skip_wait_only = transport_only && g.debug_front_half_mode == 14;

    if (threadIdx.x == 0) {
        if constexpr (C::PREQUANT_A) {
            g.A_pre.template prefetch_tma<typename G::A_fp4x2_tile>();
            g.A_pre_sc.template prefetch_tma<typename G::A_sc_tile>();
        } else {
            g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        }
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int row_block_idx = blockIdx.y * C::CLUSTER_SIZE + cta_id;
    const int col_block_idx = clusterIdx().x;
    const int local_b_col_tile_idx = clusterIdx().x * C::CLUSTER_SIZE + cta_id;
    const int num_red_blocks =
        C::PREQUANT_A ? static_cast<int>((2 * g.A_pre.cols()) / C::Kb)
                      : static_cast<int>(g.A_bf16.cols() / C::Kb);

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
    typename G::quant_buf_t (&raw_a)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::quant_buf_t, C::LOAD_PIPE_DEPTH>();
    typename G::quant_buf_t (&raw_b)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::quant_buf_t, C::LOAD_PIPE_DEPTH>();

    tensor_allocator<1, C::TMEM_NCTA, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore a_raw_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_scale_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_raw_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_scale_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_peer_amax_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float b_peer_amax[C::LOAD_PIPE_DEPTH];
    __shared__ float b_shared_amax[C::LOAD_PIPE_DEPTH];
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        #pragma unroll
        for (int s = 0; s < C::LOAD_PIPE_DEPTH; ++s) {
            init_semaphore(a_raw_arrived[s], 0, 1);
            init_semaphore(a_scale_arrived[s], 0, 1);
            init_semaphore(b_raw_arrived[s], 0, 1);
            init_semaphore(a_quant_done[s], 0, C::CLUSTER_SIZE);
            init_semaphore(b_quant_done[s], 0, C::CLUSTER_SIZE);
            init_semaphore(b_scale_arrived[s], 0, 1);
            init_semaphore(b_peer_amax_arrived[s], 0, 1);
            init_semaphore(inputs_finished[s], 0, 1);
            b_peer_amax[s] = 0.0f;
            b_shared_amax[s] = 0.0f;
        }
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (!transport_only) {
        if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        if (warpgroup_id == 0 || warpgroup_id == 3) {
            tm_allocator.set_addr(tmem_addr);
        }
    }
    everyone::tma::cluster::wait();
    __syncthreads();

    const bool a_debug_print_mode =
        transport_only &&
        (a_quant_only || a_stage1_quant_only || a_quant_per_stage_only ||
         a_quant_then_wait_only || a_quant_then_skip_wait_only);
    const bool b_debug_print_mode =
        transport_only &&
        (b_quant_only || b_stage1_quant_only || b_quant_per_stage_only ||
         b_quant_then_wait_only);
    if (warpgroup_id == 0) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        const bool is_lane_leader = (warp::laneid() == 0);
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;

        if (warp_id == 0 && is_lane_leader) {
            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                if (!transport_only) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                }
                if constexpr (C::PREQUANT_A) {
                    tma::expect(a_raw_arrived[stage], input_tiles[stage].A);
                    tma::load_async(
                        input_tiles[stage].A,
                        g.A_pre,
                        {row_block_idx, red_block_idx},
                        a_raw_arrived[stage]);
                    tma::expect(a_scale_arrived[stage], input_scales[stage].A);
                    tma::load_async(
                        input_scales[stage].A,
                        g.A_pre_sc,
                        {row_block_idx, red_block_idx, 0},
                        a_scale_arrived[stage]);
                } else {
                    tma::expect(a_raw_arrived[stage], raw_a[stage].bf16_tile);
                    tma::load_async(
                        raw_a[stage].bf16_tile,
                        g.A_bf16,
                        {row_block_idx, red_block_idx},
                        a_raw_arrived[stage]);
                }
                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        } else if (warp_id == 1 && is_lane_leader) {
            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                if (!transport_only) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                }
                tma::expect(b_raw_arrived[stage], raw_b[stage].bf16_tile);
                tma::load_async(
                    raw_b[stage].bf16_tile,
                    g.B_bf16,
                    {local_b_col_tile_idx, red_block_idx},
                    b_raw_arrived[stage]);
                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        }
        return;
    }

    if (warpgroup_id == 1) {
        if (producer_only || b_only || b_wait_only || b_quant_only || b_stage1_quant_only ||
            b_quant_per_stage_only || b_quant_then_wait_only) {
            return;
        }
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        uint32_t stage = 0;
        uint32_t ready_phasebits = 0xFFFF0000;
        uint32_t raw_phasebits = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader && !C::PREQUANT_A) {
                running_amax_a = 0.0f;
            }
            warpgroup::sync(a_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            if (!transport_only) {
                wait(inputs_finished[stage], get_phasebit<1>(ready_phasebits, stage));
            }
            const bool skip_raw_wait = a_quant_then_skip_wait_only && red_block_idx > 0;
            if (!skip_raw_wait) {
                wait(a_raw_arrived[stage], get_phasebit<0>(raw_phasebits, stage));
                if constexpr (C::PREQUANT_A) {
                    wait(a_scale_arrived[stage], get_phasebit<0>(raw_phasebits, stage));
                }
            }
            const uint32_t quant_stage = (transport_only && !a_quant_per_stage_only) ? 0 : stage;
            if (a_debug_print_mode && cta_id == 0 && is_leader && red_block_idx == 0) {
                const uint32_t *raw_a1_words =
                    reinterpret_cast<const uint32_t *>(&raw_a[1].bf16_tile);
                const uint32_t *a_raw1_sem_words =
                    reinterpret_cast<const uint32_t *>(&a_raw_arrived[1]);
                SHARED_B_DEBUG_PRINTF(
                    "[shared-b a-debug] wg1 postwait red=%d stage=%u quant_stage=%u raw1_w0=0x%08x "
                    "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                    red_block_idx, stage, quant_stage, raw_a1_words[0], raw_a1_words[1],
                    a_raw1_sem_words[0], a_raw1_sem_words[1]);
            }
            if (a_wait_only) {
                if (is_leader) {
                    arrive(a_quant_done[stage], 1);
                }
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            if (a_stage1_quant_only && red_block_idx == 0) {
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            if (a_quant_then_wait_only && red_block_idx > 0) {
                if (a_debug_print_mode && cta_id == 0 && is_leader) {
                    const uint32_t *raw_a1_words =
                        reinterpret_cast<const uint32_t *>(&raw_a[1].bf16_tile);
                    const uint32_t *a_raw1_sem_words =
                        reinterpret_cast<const uint32_t *>(&a_raw_arrived[1]);
                    SHARED_B_DEBUG_PRINTF(
                        "[shared-b a-debug] wg1 second-stage wait-only red=%d stage=%u raw1_w0=0x%08x "
                        "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                        red_block_idx, stage, raw_a1_words[0], raw_a1_words[1],
                        a_raw1_sem_words[0], a_raw1_sem_words[1]);
                }
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            if (a_quant_then_skip_wait_only && red_block_idx > 0) {
                if (a_debug_print_mode && cta_id == 0 && is_leader) {
                    const uint32_t *raw_a1_words =
                        reinterpret_cast<const uint32_t *>(&raw_a[1].bf16_tile);
                    const uint32_t *a_raw1_sem_words =
                        reinterpret_cast<const uint32_t *>(&a_raw_arrived[1]);
                    SHARED_B_DEBUG_PRINTF(
                        "[shared-b a-debug] wg1 second-stage skip-wait red=%d stage=%u raw1_w0=0x%08x "
                        "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                        red_block_idx, stage, raw_a1_words[0], raw_a1_words[1],
                        a_raw1_sem_words[0], a_raw1_sem_words[1]);
                }
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }

            if constexpr (C::PREQUANT_A) {
                warpgroup::sync(a_quant_bar_id);
            } else {
                if constexpr (C::USE_CTA_AMAX) {
                    quantize_operand_subtile_cta_amax(
                        raw_a[stage], input_tiles[quant_stage].A, input_scales[quant_stage].A, 0,
                        a_quant_bar_id, &running_amax_a, warp_max_buf_a);
                } else {
                    quantize_operand_subtile_constant(
                        raw_a[stage], input_tiles[quant_stage].A, input_scales[quant_stage].A, 0);
                }
            }
            if (a_debug_print_mode && cta_id == 0 && is_leader && red_block_idx == 0) {
                const uint32_t *raw_a1_words =
                    reinterpret_cast<const uint32_t *>(&raw_a[1].bf16_tile);
                const uint32_t *a_raw1_sem_words =
                    reinterpret_cast<const uint32_t *>(&a_raw_arrived[1]);
                SHARED_B_DEBUG_PRINTF(
                    "[shared-b a-debug] wg1 postquant red=%d stage=%u quant_stage=%u raw1_w0=0x%08x "
                    "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                    red_block_idx, stage, quant_stage, raw_a1_words[0], raw_a1_words[1],
                    a_raw1_sem_words[0], a_raw1_sem_words[1]);
            }
            if (transport_only && (a_quant_only || a_stage1_quant_only || a_quant_per_stage_only ||
                                   a_quant_then_wait_only || a_quant_then_skip_wait_only)) {
                warpgroup::sync(a_quant_bar_id);
                asm volatile("" ::: "memory");
            }
            if (a_quant_only || a_stage1_quant_only || a_quant_per_stage_only ||
                a_quant_then_wait_only || a_quant_then_skip_wait_only) {
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            warpgroup::sync(a_quant_bar_id);
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

            if (is_leader) {
                if (cta_id == 0) {
                    arrive(a_quant_done[stage], 1);
                } else {
                    tma::cluster::arrive(a_quant_done[stage], 0, 1);
                }
            }

            update_phasebit<1>(ready_phasebits, stage);
            update_phasebit<0>(raw_phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
        return;
    }

    if (warpgroup_id == 2) {
        if (producer_only || a_only || a_wait_only || a_quant_only || a_stage1_quant_only ||
            a_quant_per_stage_only || a_quant_then_wait_only || a_quant_then_skip_wait_only) {
            return;
        }
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int b_quant_bar_id = 3;
        uint32_t stage = 0;
        uint32_t ready_phasebits = 0xFFFF0000;
        uint32_t raw_phasebits = 0;
        uint32_t amax_phasebits = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_b = 0.0f;
            }
            warpgroup::sync(b_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            if (!transport_only) {
                wait(inputs_finished[stage], get_phasebit<1>(ready_phasebits, stage));
            }
            wait(b_raw_arrived[stage], get_phasebit<0>(raw_phasebits, stage));
            const uint32_t quant_stage = (transport_only && !b_quant_per_stage_only) ? 0 : stage;
            if (b_debug_print_mode && is_leader &&
                (red_block_idx == 0 || red_block_idx == 1)) {
                const uint32_t *raw_b1_words =
                    reinterpret_cast<const uint32_t *>(&raw_b[1].bf16_tile);
                const uint32_t *b_raw1_sem_words =
                    reinterpret_cast<const uint32_t *>(&b_raw_arrived[1]);
                SHARED_B_DEBUG_PRINTF(
                    "[shared-b b-debug] cta=%d wg2 postwait red=%d stage=%u quant_stage=%u raw1_w0=0x%08x "
                    "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                    cta_id, red_block_idx, stage, quant_stage, raw_b1_words[0], raw_b1_words[1],
                    b_raw1_sem_words[0], b_raw1_sem_words[1]);
            }
            if (b_wait_only) {
                if (is_leader) {
                    arrive(b_quant_done[stage], 1);
                }
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                update_phasebit<0>(amax_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            if (b_stage1_quant_only && red_block_idx == 0) {
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                update_phasebit<0>(amax_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            if (b_quant_then_wait_only && red_block_idx > 0) {
                if (b_debug_print_mode && is_leader) {
                    const uint32_t *raw_b1_words =
                        reinterpret_cast<const uint32_t *>(&raw_b[1].bf16_tile);
                    const uint32_t *b_raw1_sem_words =
                        reinterpret_cast<const uint32_t *>(&b_raw_arrived[1]);
                    SHARED_B_DEBUG_PRINTF(
                        "[shared-b b-debug] cta=%d wg2 second-stage wait-only red=%d stage=%u raw1_w0=0x%08x "
                        "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                        cta_id, red_block_idx, stage, raw_b1_words[0], raw_b1_words[1],
                        b_raw1_sem_words[0], b_raw1_sem_words[1]);
                }
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                update_phasebit<0>(amax_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }

            if constexpr (C::USE_CTA_AMAX) {
                const float local_b_amax = compute_operand_subtile_amax(
                    raw_b[stage], 0, b_quant_bar_id, warp_max_buf_b);
                if (is_leader) {
                    if (!transport_only && red_block_idx < 3) {
                        SHARED_B_DEBUG_PRINTF("[shared-b cta%d wg2 red=%d stage=%u pre-peer local=%f]\n",
                               cta_id, red_block_idx, stage, local_b_amax);
                    }
                    const int peer_cta = cta_id ^ 1;
                    b_shared_amax[stage] = local_b_amax;
                    union {
                        float f;
                        uint32_t u;
                    } peer_bits{local_b_amax};
                    const uint32_t peer_amax_addr =
                        static_cast<uint32_t>(__cvta_generic_to_shared(&b_peer_amax[stage]));
                    nvfp4_fused_gemm::st_shared_cluster_b32(
                        nvfp4_fused_gemm::map_cluster_addr(peer_amax_addr, peer_cta),
                        peer_bits.u);
                    nvfp4_fused_gemm::arrive_remote_cluster(
                        b_peer_amax_arrived[stage], peer_cta, 1);
                    wait(b_peer_amax_arrived[stage], get_phasebit<0>(amax_phasebits, stage));
                    asm volatile("" ::: "memory");
                    if (!transport_only && red_block_idx < 3) {
                        SHARED_B_DEBUG_PRINTF("[shared-b cta%d wg2 red=%d stage=%u post-peer peer=%f]\n",
                               cta_id, red_block_idx, stage, b_peer_amax[stage]);
                    }
                    b_shared_amax[stage] = fmaxf(local_b_amax, b_peer_amax[stage]);
                    running_amax_b = fmaxf(running_amax_b, b_shared_amax[stage]);
                }
                warpgroup::sync(b_quant_bar_id);
                quantize_operand_subtile_cta_amax_with_global(
                    raw_b[stage], input_tiles[quant_stage].B[0], input_scales[quant_stage].B[cta_id], 0,
                    b_shared_amax[stage], &running_amax_b);
                if (!transport_only && is_leader && red_block_idx < 3) {
                    SHARED_B_DEBUG_PRINTF("[shared-b cta%d wg2 red=%d stage=%u post-quant shared=%f running=%f]\n",
                           cta_id, red_block_idx, stage, b_shared_amax[stage], running_amax_b);
                }
            } else {
                quantize_operand_subtile_constant(
                    raw_b[stage], input_tiles[quant_stage].B[0], input_scales[quant_stage].B[cta_id], 0);
            }
            if (b_debug_print_mode && is_leader &&
                (red_block_idx == 0 || red_block_idx == 1)) {
                const uint32_t *raw_b1_words =
                    reinterpret_cast<const uint32_t *>(&raw_b[1].bf16_tile);
                const uint32_t *b_raw1_sem_words =
                    reinterpret_cast<const uint32_t *>(&b_raw_arrived[1]);
                SHARED_B_DEBUG_PRINTF(
                    "[shared-b b-debug] cta=%d wg2 postquant red=%d stage=%u quant_stage=%u raw1_w0=0x%08x "
                    "raw1_w1=0x%08x sem1_w0=0x%08x sem1_w1=0x%08x\n",
                    cta_id, red_block_idx, stage, quant_stage, raw_b1_words[0], raw_b1_words[1],
                    b_raw1_sem_words[0], b_raw1_sem_words[1]);
            }
            if (transport_only && (b_quant_only || b_stage1_quant_only || b_quant_per_stage_only ||
                                   b_quant_then_wait_only)) {
                warpgroup::sync(b_quant_bar_id);
                asm volatile("" ::: "memory");
            }
            if (b_quant_only || b_stage1_quant_only || b_quant_per_stage_only || b_quant_then_wait_only) {
                if (b_debug_print_mode && is_leader &&
                    (red_block_idx == 0 || red_block_idx == 1)) {
                    SHARED_B_DEBUG_PRINTF(
                        "[shared-b b-debug] cta=%d wg2 precontinue red=%d stage=%u ready_phase=0x%08x raw_phase=0x%08x "
                        "amax_phase=0x%08x\n",
                        cta_id, red_block_idx, stage, ready_phasebits, raw_phasebits, amax_phasebits);
                }
                update_phasebit<1>(ready_phasebits, stage);
                update_phasebit<0>(raw_phasebits, stage);
                update_phasebit<0>(amax_phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                continue;
            }
            warpgroup::sync(b_quant_bar_id);

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

            if (!transport_only && is_leader) {
                const int peer_cta = cta_id ^ 1;
                tma::cluster::expect_bytes(b_scale_arrived[stage], sizeof(typename G::B_sc_tile), peer_cta);
                tma::cluster::store_async(
                    reinterpret_cast<void *>(&input_scales[stage].B[cta_id]),
                    reinterpret_cast<void *>(&input_scales[stage].B[cta_id]),
                    sizeof(typename G::B_sc_tile),
                    peer_cta,
                    b_scale_arrived[stage]);
            }

            if (is_leader) {
                if (cta_id == 0) {
                    arrive(b_quant_done[stage], 1);
                } else {
                    tma::cluster::arrive(b_quant_done[stage], 0, 1);
                }
            }

            update_phasebit<1>(ready_phasebits, stage);
            update_phasebit<0>(raw_phasebits, stage);
            update_phasebit<0>(amax_phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
        if (b_debug_print_mode && is_leader) {
            SHARED_B_DEBUG_PRINTF("[shared-b b-debug] cta=%d wg2 exit\n", cta_id);
        }
        return;
    }

    if (warpgroup_id == 3) {
        if (transport_only) {
            return;
        }
        if (warp::elect_leader()) {
            const int warp_id = group<WARPGROUP_WARPS>::warpid();
            if (cta_id == 0 && warp_id == 0) {
                auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                uint32_t stage = 0;
                uint32_t mma_phasebits = 0;
                uint32_t scale_phasebits = 0;

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    tma::cluster::wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    tma::cluster::wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    if constexpr (C::USE_CTA_AMAX) {
                        if (red_block_idx < 3) {
                            SHARED_B_DEBUG_PRINTF("[shared-b wg3 red=%d stage=%u post-quant-waits]\n",
                                   red_block_idx, stage);
                        }
                    }
                    {
                        wait(b_scale_arrived[stage], get_phasebit<0>(scale_phasebits, stage));
                    }
                    if constexpr (C::USE_CTA_AMAX) {
                        if (red_block_idx < 3) {
                            SHARED_B_DEBUG_PRINTF("[shared-b wg3 red=%d stage=%u post-scale-wait]\n",
                                   red_block_idx, stage);
                        }
                    }

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile =
                            A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile_0 =
                            B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * 32);
                        auto &B_sc_sm_subtile_0 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                        auto B_sc_tm_subtile_1 =
                            B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * 32 + 16);
                        auto &B_sc_sm_subtile_1 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");

                    if (red_block_idx == 0) {
                        mm2_ABt(
                            out_tm, input_tiles[stage].A, input_tiles[stage].B[0],
                            A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]);
                    } else {
                        mma2_ABt(
                            out_tm, input_tiles[stage].A, input_tiles[stage].B[0],
                            A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                            B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                            inputs_finished[stage]);
                    }

                    mma_phasebits ^= (1u << stage);
                    update_phasebit<0>(scale_phasebits, stage);
                    if constexpr (C::USE_CTA_AMAX) {
                        if (red_block_idx < 3) {
                            SHARED_B_DEBUG_PRINTF("[shared-b wg3 red=%d stage=%u post-mma]\n",
                                   red_block_idx, stage);
                        }
                    }
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived);
            }
        }

        wait(outputs_arrived, 0);

        const float a_sg_dec = C::PREQUANT_A
                                   ? g.A_pre_sg[{0}]
                                   : (C::USE_CTA_AMAX ? running_amax_a / (6.0f * 448.0f) : SCALE_MAX_DEC);
        const float b_sg_dec = C::USE_CTA_AMAX ? running_amax_b / (6.0f * 448.0f) : SCALE_MAX_DEC;
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
            warpgroup::load_async(
                D_reg_fl,
                out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                    0, C::Nb / C::EPI_PIPE_DEPTH * epi));
            warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
            warp::copy(D_reg[epi], D_reg_fl);
        }

        tensor_load_wait();
        tensor_before_thread_sync();
        warpgroup::sync(1);
        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

        #pragma unroll
        for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
            warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
            warpgroup::sync(1);
            warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
            warpgroup::sync(1);
            warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                g.D,
                output_tiles.D[epi % C::NUM_D_TILES],
                {row_block_idx, C::EPI_PIPE_DEPTH * col_block_idx + epi});
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    if constexpr (C::CLUSTER_REUSE_2D) {
        kernel_cluster_reuse_2d<C>(g);
        return;
    }

    if constexpr (C::SHARE_B_ACROSS_CTAS) {
        kernel_shared_b_cross_cta<C>(g);
        return;
    }

    if constexpr (C::SHARE_A_ACROSS_CTAS) {
        kernel_shared_a_cross_cta<C>(g);
        return;
    }
    
    if constexpr (C::DEDICATED_PRODUCER && !C::SHARE_B_ACROSS_CTAS) {
        static_assert(C::COL_TILES_PER_BLOCK == 1,
                      "Dedicated-producer backend currently supports one column tile per CTA");

        if (threadIdx.x == 0) {
            g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
            g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
            g.D.template prefetch_tma<typename G::D_tile>();
        }

        const int warpgroup_id = warpgroup::groupid();
        const int row_block_idx = blockIdx.y;
        const int col_block_idx = blockIdx.x;
        const int num_red_blocks = g.A_bf16.cols() / C::Kb;

        extern __shared__ int __shm[];
        tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
        typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
            sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
        typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
            sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
        typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
        typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
        typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();

        tensor_allocator<1, C::TMEM_NCTA, false> tm_allocator;

        __shared__ uint32_t tmem_addr;
        __shared__ semaphore tmem_provisioned;
        __shared__ semaphore outputs_arrived;
        __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
        __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
        __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
        __shared__ semaphore a_raw_arrived;
        __shared__ semaphore a_raw_consumed;
        __shared__ semaphore b_raw_arrived;
        __shared__ semaphore b_raw_consumed;
        __shared__ float running_amax_a;
        __shared__ float running_amax_b;
        __shared__ float warp_max_buf_a[4];
        __shared__ float warp_max_buf_b[4];

        if (threadIdx.x == 32) {
            init_semaphore(tmem_provisioned, 0, 1);
            init_semaphore(outputs_arrived, 0, 1);
            init_semaphore(a_raw_arrived, 0, 1);
            init_semaphore(a_raw_consumed, 0, 1);
            init_semaphore(b_raw_arrived, 0, 1);
            init_semaphore(b_raw_consumed, 0, 1);
            #pragma unroll
            for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
                init_semaphore(a_quant_done[stage], 0, 1);
                init_semaphore(b_quant_done[stage], 0, 1);
                init_semaphore(inputs_finished[stage], 0, 1);
            }
            if constexpr (C::USE_CTA_AMAX) {
                running_amax_a = 0.0f;
                running_amax_b = 0.0f;
            }
        }
        everyone::tma::cluster::arrive_aligned();

        if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        if (warpgroup_id == 3) {
            tm_allocator.set_addr(tmem_addr);
        }
        everyone::tma::cluster::wait();
        __syncthreads();

        if (warpgroup_id == 0) {
            const int warp_id = group<WARPGROUP_WARPS>::warpid();
            const bool is_lane_leader = (warp::laneid() == 0);

            if (warp_id == 0 && is_lane_leader) {
                int a_raw_consumed_phase = 0;
                bool first_a = true;
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    #pragma unroll
                    for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                        if (!first_a) {
                            wait(a_raw_consumed, a_raw_consumed_phase);
                            a_raw_consumed_phase ^= 1;
                        }

                        tma::expect(a_raw_arrived, quant_buf_a.bf16_tile);
                        tma::load_async(
                            quant_buf_a.bf16_tile,
                            g.A_bf16,
                            {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            a_raw_arrived);
                        first_a = false;
                    }
                }
            } else if (warp_id == 1 && is_lane_leader) {
                int b_raw_consumed_phase = 0;
                bool first_b = true;
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    #pragma unroll
                    for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                        if (!first_b) {
                            wait(b_raw_consumed, b_raw_consumed_phase);
                            b_raw_consumed_phase ^= 1;
                        }

                        tma::expect(b_raw_arrived, quant_buf_b.bf16_tile);
                        tma::load_async(
                            quant_buf_b.bf16_tile,
                            g.B_bf16,
                            {col_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                            b_raw_arrived);
                        first_b = false;
                    }
                }
            }
        } else if (warpgroup_id == 1) {
            const int local_tid = threadIdx.x % 128;
            const bool is_leader = (local_tid == 0);
            constexpr int a_quant_bar_id = 2;
            uint32_t stage = 0;
            uint32_t phasebits = 0xFFFF0000;
            int raw_phase = 0;

            if constexpr (C::USE_CTA_AMAX) {
                if (is_leader) {
                    running_amax_a = 0.0f;
                }
                warpgroup::sync(a_quant_bar_id);
            }

            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    wait(a_raw_arrived, raw_phase);
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                            a_quant_bar_id, &running_amax_a, warp_max_buf_a);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub);
                    }
                    warpgroup::sync(a_quant_bar_id);
                    if (is_leader) {
                        arrive(a_raw_consumed, 1);
                    }
                    raw_phase ^= 1;
                }

                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    arrive(a_quant_done[stage], 1);
                }

                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        } else if (warpgroup_id == 2) {
            const int local_tid = threadIdx.x % 128;
            const bool is_leader = (local_tid == 0);
            constexpr int b_quant_bar_id = 3;
            uint32_t stage = 0;
            uint32_t phasebits = 0xFFFF0000;
            int raw_phase = 0;

            if constexpr (C::USE_CTA_AMAX) {
                if (is_leader) {
                    running_amax_b = 0.0f;
                }
                warpgroup::sync(b_quant_bar_id);
            }

            for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    wait(b_raw_arrived, raw_phase);
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub,
                            b_quant_bar_id, &running_amax_b, warp_max_buf_b);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_b, input_tiles[stage].B[0], input_scales[stage].B[0], sub);
                    }
                    warpgroup::sync(b_quant_bar_id);
                    if (is_leader) {
                        arrive(b_raw_consumed, 1);
                    }
                    raw_phase ^= 1;
                }

                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (is_leader) {
                    arrive(b_quant_done[stage], 1);
                }

                update_phasebit<1>(phasebits, stage);
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        } else if (warpgroup_id == 3) {
            if (warp::elect_leader()) {
                const int warp_id = group<WARPGROUP_WARPS>::warpid();
                if (warp_id == 0) {
                    auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                    auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
                    auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                        256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                    uint32_t stage = 0;
                    uint32_t mma_phasebits = 0;

                    tensor_after_thread_sync();
                    for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                        wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                        wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                        #pragma unroll
                        for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                            auto &A_sc_sm_subtile =
                                *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                    reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                            auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                            auto &B_sc_sm_subtile =
                                *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                    reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                        }

                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                        auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                        if (red_block_idx == 0) {
                            mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                        } else {
                            mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                        }
                        kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                        mma_phasebits ^= (1u << stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                    tensor_commit<1>(outputs_arrived);
                }
            }

            wait(outputs_arrived, 0);
            const float a_sg_dec = C::USE_CTA_AMAX ? running_amax_a / (6.0f * 448.0f) : SCALE_MAX_DEC;
            const float b_sg_dec = C::USE_CTA_AMAX ? running_amax_b / (6.0f * 448.0f) : SCALE_MAX_DEC;
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

            rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                warpgroup::load_async(
                    D_reg_fl,
                    out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                        0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                warp::copy(D_reg[epi], D_reg_fl);
            }

            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);

            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                    g.D, output_tiles.D[epi % C::NUM_D_TILES],
                    {row_block_idx, C::EPI_PIPE_DEPTH * col_block_idx + epi});
            }

            warpgroup::sync(1);
            warpgroup::tma::store_async_read_wait<0>();
            if (warpgroup::warpid() == 0) {
                tm_allocator.deprovision();
            }
        }
        return;
    }

    if constexpr (!C::SHARE_B_ACROSS_CTAS) {
    if (threadIdx.x == 0) {
        g.A_bf16.template prefetch_tma<typename G::A_bf16_tile>();
        g.B_bf16.template prefetch_tma<typename G::B_bf16_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int row_block_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;
    const int num_red_blocks = g.A_bf16.cols() / C::Kb;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<typename G::outputs_t>();
    typename G::quant_buf_t &quant_buf_a = sm_allocator.allocate<typename G::quant_buf_t>();
    typename G::quant_buf_t &quant_buf_b = sm_allocator.allocate<typename G::quant_buf_t>();

    tensor_allocator<1, C::TMEM_NCTA, false> tm_allocator;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned;
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore a_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore b_quant_done[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore a_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ semaphore b_bf16_sub_arrived[C::QUANT_SUB_TILES];
    __shared__ float running_amax_a;
    __shared__ float running_amax_b;
    __shared__ float warp_max_buf_a[4];
    __shared__ float warp_max_buf_b[4];

    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int stage = 0; stage < C::LOAD_PIPE_DEPTH; ++stage) {
            init_semaphore(a_quant_done[stage], 0, 1);
            init_semaphore(b_quant_done[stage], 0, 1);
            init_semaphore(inputs_finished[stage], 0, 1);
        }
        #pragma unroll
        for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
            init_semaphore(a_bf16_sub_arrived[sub], 0, 1);
            init_semaphore(b_bf16_sub_arrived[sub], 0, 1);
        }
        if constexpr (C::USE_CTA_AMAX) {
            running_amax_a = 0.0f;
            running_amax_b = 0.0f;
        }
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id == 0 && warpgroup::warpid() == 0) {
        tm_allocator.provision(tmem_addr);
        warp::arrive(tmem_provisioned);
    }
    wait(tmem_provisioned, 0);
    if (warpgroup_id == 0 || warpgroup_id == 3) {
        tm_allocator.set_addr(tmem_addr);
    }
    everyone::tma::cluster::wait();
    __syncthreads();

    if (warpgroup_id == 1) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int a_quant_bar_id = 2;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_a = 0.0f;
            }
            warpgroup::sync(a_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            #pragma unroll
            for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                if (is_leader) {
                    tma::expect(a_bf16_sub_arrived[sub], quant_buf_a.bf16_tile);
                    tma::load_async(
                        quant_buf_a.bf16_tile, g.A_bf16,
                        {row_block_idx, red_block_idx * C::QUANT_SUB_TILES + sub},
                        a_bf16_sub_arrived[sub]);
                }
                wait(a_bf16_sub_arrived[sub], sub_phase);
                if constexpr (C::USE_CTA_AMAX) {
                    quantize_operand_subtile_cta_amax(
                        quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub,
                        a_quant_bar_id, &running_amax_a, warp_max_buf_a);
                } else {
                    quantize_operand_subtile_constant(
                        quant_buf_a, input_tiles[stage].A, input_scales[stage].A, sub);
                }
                warpgroup::sync(a_quant_bar_id);
            }
            sub_phase ^= 1;

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (is_leader) {
                arrive(a_quant_done[stage], 1);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if (warpgroup_id == 2) {
        const int local_tid = threadIdx.x % 128;
        const bool is_leader = (local_tid == 0);
        constexpr int b_quant_bar_id = 3;
        uint32_t stage = 0;
        uint32_t phasebits = 0xFFFF0000;
        int sub_phase = 0;

        if constexpr (C::USE_CTA_AMAX) {
            if (is_leader) {
                running_amax_b = 0.0f;
            }
            warpgroup::sync(b_quant_bar_id);
        }

        for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
            #pragma unroll
            for (int col_tile = 0; col_tile < C::COL_TILES_PER_BLOCK; ++col_tile) {
                #pragma unroll
                for (int sub = 0; sub < C::QUANT_SUB_TILES; ++sub) {
                    if (is_leader) {
                        tma::expect(b_bf16_sub_arrived[sub], quant_buf_b.bf16_tile);
                        tma::load_async(
                            quant_buf_b.bf16_tile, g.B_bf16,
                            {col_block_idx * C::COL_TILES_PER_BLOCK + col_tile,
                             red_block_idx * C::QUANT_SUB_TILES + sub},
                            b_bf16_sub_arrived[sub]);
                    }
                    wait(b_bf16_sub_arrived[sub], sub_phase);
                    if constexpr (C::USE_CTA_AMAX) {
                        quantize_operand_subtile_cta_amax(
                            quant_buf_b, input_tiles[stage].B[col_tile], input_scales[stage].B[col_tile], sub,
                            b_quant_bar_id, &running_amax_b, warp_max_buf_b);
                    } else {
                        quantize_operand_subtile_constant(
                            quant_buf_b, input_tiles[stage].B[col_tile], input_scales[stage].B[col_tile], sub);
                    }
                    warpgroup::sync(b_quant_bar_id);
                }
                sub_phase ^= 1;
            }

            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (is_leader) {
                arrive(b_quant_done[stage], 1);
            }

            update_phasebit<1>(phasebits, stage);
            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
        }
    } else if (warpgroup_id == 3 && warp::elect_leader()) {
        const int warp_id = group<WARPGROUP_WARPS>::warpid();
        if (warp_id == 0) {
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            uint32_t stage = 0;
            uint32_t mma_phasebits = 0;

            if constexpr (C::COL_TILES_PER_BLOCK == 1) {
                auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile, B_sc_sm_subtile);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile = B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    if (red_block_idx == 0) {
                        mm_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                    } else {
                        mma_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                    mma_phasebits ^= (1u << stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
            } else if constexpr (C::COL_TILES_PER_BLOCK == 2) {
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    256 + 8 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_0 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                        auto B_sc_tm_subtile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_1 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    if (red_block_idx == 0) {
                        mm_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mm_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                    } else {
                        mma_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mma_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                    mma_phasebits ^= (1u << stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
            } else {
                auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
                auto out_tm_2 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(256);
                auto out_tm_3 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(384);
                auto B_sc_tm_0 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_1 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 8 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_2 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 12 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
                auto B_sc_tm_3 = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(
                    512 + 16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

                tensor_after_thread_sync();
                for (int red_block_idx = 0; red_block_idx < num_red_blocks; ++red_block_idx) {
                    wait(a_quant_done[stage], (mma_phasebits >> stage) & 0x1);
                    wait(b_quant_done[stage], (mma_phasebits >> stage) & 0x1);

                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &A_sc_sm_subtile =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(A_sc_tm_subtile, A_sc_sm_subtile);

                        auto B_sc_tm_subtile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_0 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                        auto B_sc_tm_subtile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_1 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_1, B_sc_sm_subtile_1);

                        auto B_sc_tm_subtile_2 = B_sc_tm_2.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_2 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[2].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_2, B_sc_sm_subtile_2);

                        auto B_sc_tm_subtile_3 = B_sc_tm_3.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                        auto &B_sc_sm_subtile_3 =
                            *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[3].data[0]) + 16 * 32 * ii);
                        load_mxnv_scale_async<1>(B_sc_tm_subtile_3, B_sc_sm_subtile_3);
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    auto A_sc_tm_tile = A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_0 = B_sc_tm_0.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_1 = B_sc_tm_1.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_2 = B_sc_tm_2.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_tm_tile_3 = B_sc_tm_3.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    if (red_block_idx == 0) {
                        mm_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mm_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                        mm_ABt(out_tm_2, input_tiles[stage].A, input_tiles[stage].B[2], A_sc_tm_tile, B_sc_tm_tile_2);
                        mm_ABt(out_tm_3, input_tiles[stage].A, input_tiles[stage].B[3], A_sc_tm_tile, B_sc_tm_tile_3);
                    } else {
                        mma_ABt(out_tm_0, input_tiles[stage].A, input_tiles[stage].B[0], A_sc_tm_tile, B_sc_tm_tile_0);
                        mma_ABt(out_tm_1, input_tiles[stage].A, input_tiles[stage].B[1], A_sc_tm_tile, B_sc_tm_tile_1);
                        mma_ABt(out_tm_2, input_tiles[stage].A, input_tiles[stage].B[2], A_sc_tm_tile, B_sc_tm_tile_2);
                        mma_ABt(out_tm_3, input_tiles[stage].A, input_tiles[stage].B[3], A_sc_tm_tile, B_sc_tm_tile_3);
                    }
                    kittens::detail::tcgen05::commit<1>(inputs_finished[stage]);

                    mma_phasebits ^= (1u << stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<1>(outputs_arrived);
            }
        }
    }

    if (warpgroup_id == 0) {
        wait(outputs_arrived, 0);
        const float a_sg_dec = C::USE_CTA_AMAX ? running_amax_a / (6.0f * 448.0f) : SCALE_MAX_DEC;
        const float b_sg_dec = C::USE_CTA_AMAX ? running_amax_b / (6.0f * 448.0f) : SCALE_MAX_DEC;
        if constexpr (C::COL_TILES_PER_BLOCK == 1) {
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

            rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                warpgroup::load_async(
                    D_reg_fl,
                    out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                        0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                warp::copy(D_reg[epi], D_reg_fl);
            }

            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);

            #pragma unroll
            for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                    g.D, output_tiles.D[epi % C::NUM_D_TILES],
                    {row_block_idx, C::EPI_PIPE_DEPTH * col_block_idx + epi});
            }
        } else if constexpr (C::COL_TILES_PER_BLOCK == 2) {
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);

            auto store_dual_tile = [&](auto &out_tm_sel, int col_tile_offset) {
                rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(
                        D_reg_fl,
                        out_tm_sel.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                            0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                    warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                    warp::copy(D_reg[epi], D_reg_fl);
                }

                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                        g.D, output_tiles.D[epi % C::NUM_D_TILES],
                        {row_block_idx,
                         C::EPI_PIPE_DEPTH * (col_block_idx * C::COL_TILES_PER_BLOCK + col_tile_offset) + epi});
                }
            };

            store_dual_tile(out_tm_0, 0);
            store_dual_tile(out_tm_1, 1);
        } else {
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto out_tm_2 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(256);
            auto out_tm_3 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(384);

            auto store_quad_tile = [&](auto &out_tm_sel, int col_tile_offset) {
                rt_bf<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    rt_fl<C::Mb / 4, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(
                        D_reg_fl,
                        out_tm_sel.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                            0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                    warp::mul(D_reg_fl, D_reg_fl, a_sg_dec * b_sg_dec);
                    warp::copy(D_reg[epi], D_reg_fl);
                }

                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[epi % C::NUM_D_TILES], D_reg[epi]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                        g.D, output_tiles.D[epi % C::NUM_D_TILES],
                        {row_block_idx,
                         C::EPI_PIPE_DEPTH * (col_block_idx * C::COL_TILES_PER_BLOCK + col_tile_offset) + epi});
                }
            };

            store_quad_tile(out_tm_0, 0);
            store_quad_tile(out_tm_1, 1);
            store_quad_tile(out_tm_2, 2);
            store_quad_tile(out_tm_3, 3);
        }

        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) {
            tm_allocator.deprovision();
        }
    }
    }
}

} // namespace nvfp4_fused_gemm_both_bf16
