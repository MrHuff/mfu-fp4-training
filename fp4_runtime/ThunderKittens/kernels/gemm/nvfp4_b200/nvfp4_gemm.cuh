#pragma once
// ================================================================
// NVFP4 Standard GEMM Kernel
// Single GEMM: D = A × B^T with optional QKV split output.
// ================================================================

#include "kittens.cuh"
#include "nvfp4_rope_epilogue.cuh"
#include "../common/h_nvfp4_tile_carrier.cuh"

using namespace kittens;

namespace nvfp4_gemm {

template <
    int _Nb,
    int _LOAD_PIPE_DEPTH,
    int _EPI_PIPE_DEPTH,
    int _SUPERGROUP_SIZE,
    int _NUM_D_TILES,
    bool _OVERLAP_EPI,
    int _Mb = 256,
    bool _USE_PDL = true,
    int _CLUSTER_SIZE = 2,
    int _Kb = 256,
    bool _ROPE_LIVE64 = false,
    bool _FUSE_RESIDUAL = false,
    bool _FUSE_H_NV_CARRIER = false,
    bool _FUSE_C1_RMS = false,
    bool _FUSE_C1_RMS_CTA = false,
    bool _PIPELINE_RESIDUAL = false
>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_Mb == 256 || _Mb == 512, "Mb must be 256 or 512");
    static_assert(_Kb == 128 || _Kb == 256, "Kb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5, "LOAD_PIPE_DEPTH must be greater than 0 and at most 5");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be greater than 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be greater than 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be greater than 0");
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2, "NUM_D_TILES must be at least 2 if EPI_PIPE_DEPTH > 1");

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
    static constexpr int Kb = _Kb;
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int MMA_PER_TILE = Kb/64;

    static constexpr int NUM_D_TILES = _NUM_D_TILES;
    static constexpr bool ROPE_LIVE64 = _ROPE_LIVE64;
    static constexpr bool FUSE_RESIDUAL = _FUSE_RESIDUAL;
    static constexpr bool FUSE_H_NV_CARRIER = _FUSE_H_NV_CARRIER;
    static constexpr bool FUSE_C1_RMS = _FUSE_C1_RMS;
    static constexpr bool FUSE_C1_RMS_CTA = _FUSE_C1_RMS_CTA;
    static constexpr bool PIPELINE_RESIDUAL = _PIPELINE_RESIDUAL;
    static_assert(
        !FUSE_H_NV_CARRIER ||
            (FUSE_RESIDUAL && !_OVERLAP_EPI && _Nb == 256 &&
             _EPI_PIPE_DEPTH == 8),
        "H NV carrier requires residual and 256 columns in eight slices");
    static_assert(!(FUSE_H_NV_CARRIER && FUSE_C1_RMS),
                  "exact row RMS and tile RMS carriers are distinct epilogues");
    static_assert(
        !FUSE_C1_RMS_CTA ||
            (FUSE_C1_RMS && FUSE_RESIDUAL && !OVERLAP_EPI),
        "CTA row RMS reduction requires the non-overlapped residual epilogue");
    static_assert(
        !PIPELINE_RESIDUAL ||
            (FUSE_RESIDUAL && !OVERLAP_EPI && !FUSE_H_NV_CARRIER),
        "residual prefetch pipeline requires the ordinary non-overlapped residual epilogue");

    // Output cache policy for TMA stores
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;
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

template <typename Tile, bool Enabled>
struct residual_tiles_storage {};

template <typename Tile>
struct residual_tiles_storage<Tile, true> {
    Tile R[2];
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float,      1,  1,  1,  1>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl     A;           // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (N // 64) x 256
    A_sc_global_gl A_sc_global; // (1,)
    B_fp4x2_gl     B;           // M x (N // 2)
    B_sc_gl        B_sc;        // (M // 128) x (N // 64) x 256
    B_sc_global_gl B_sc_global; // (1,)
    D_gl           D;           // M x N

    // Optional independent outputs for QKV grouped GEMM
    D_gl           D_K;         // M x N_k
    D_gl           D_V;         // M x N_v
    int            q_dim;       // Columns for Q
    int            k_dim;       // Columns for K
    int            v_dim;       // Columns for V
    bool           use_split_D; // Trigger manual splitting logic

    // Optional per-row/per-col outer scales (raw pointers, not TMA).
    // nullptr = use the scalar globals for that operand.
    const float* a_sg_per_tile;
    int          a_sg_stride;
    const float* b_sg_per_tile;
    int          b_sg_stride;
    const float* a_sg_chunk_grid;
    int          a_sg_chunk_stride;
    const float* b_sg_chunk_grid;
    int          b_sg_chunk_stride;

    // SiLU epilogue: apply silu(x) = x * sigmoid(x) to output columns [0, silu_dim).
    // 0 = disabled. Used for SwiGLU FFN where W1 output needs SiLU.
    int silu_dim;

    // Optional RoPE epilogue. Used for Q/K split-output QKV forward only.
    nvfp4_rope_epilogue::rope_desc rope;
    nvfp4_rope_epilogue::rope_live64_desc rope_live64;
    CUtensorMap R_tma;   // optional residual descriptor, M x N
    float* h_r_tile;               // inverse RMS [M/128,N/128]
    float* h_amax_tile;            // normalized BF16 amax [M/128,N/128]
    const bf16* h_gamma;           // post-tile-RMS gamma [N]
    int h_cols;
    float h_eps;
    float* row_rms_partial;        // BF16-output row sumsq partials
    int row_rms_partial_stride;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A;
        B_sc_tile B[C::B_SC_SIZE];
    };
    struct outputs_t : residual_tiles_storage<D_tile, C::PIPELINE_RESIDUAL> {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        int d_cols = use_split_D ? (q_dim + k_dim + v_dim) : D.cols();
        return dim3(min((D.rows()/(C::Mb/2))*(d_cols/C::Nb), num_sms()));
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

// Apply SiLU activation element-wise on an rt_fl register tile.
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
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

template <typename RT>
__device__ inline void add_scaled_inplace(RT &dst, const RT &src, float scale) {
    #pragma unroll
    for (int i = 0; i < RT::height; i++) {
        #pragma unroll
        for (int j = 0; j < RT::width; j++) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k++) {
                auto &d = dst.tiles[i][j].data[k];
                const auto &s = src.tiles[i][j].data[k];
                d.x = __fmaf_rn(s.x, scale, d.x);
                d.y = __fmaf_rn(s.y, scale, d.y);
            }
        }
    }
}

template <typename C, typename RT>
__device__ inline void apply_rope_live64_if_enabled(
    RT &D_reg,
    const globals<C> &g,
    int row_block_idx,
    int cta_id,
    int col_offset_elems
) {
    if constexpr (C::ROPE_LIVE64) {
        if (g.rope.enabled()) {
            if (col_offset_elems >= g.q_dim + g.k_dim) {
                return;
            }
            nvfp4_rope_epilogue::apply_inplace(
                D_reg,
                g.rope,
                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                col_offset_elems
            );
            return;
        }
        if (!g.rope_live64.enabled()) {
            return;
        }
        if (col_offset_elems >= g.q_dim + g.k_dim) {
            return;
        }
        nvfp4_rope_epilogue::apply_inplace_live64(
            D_reg,
            g.rope_live64,
            (row_block_idx * 2 + cta_id) * (C::Mb / 2),
            col_offset_elems
        );
    }
}

template <int ROWS, int COLS>
__device__ inline void scale_shared_fp8_tile(st_fp8e4m3<ROWS, COLS, false> &tile, float scale) {
    auto *vals = reinterpret_cast<fp8e4m3*>(&tile.data[0]);
    const int lane = threadIdx.x % WARP_THREADS;
    constexpr int kPackElems = 4;
    constexpr int kNumPacks = (ROWS * COLS) / kPackElems;
    static_assert((ROWS * COLS) % kPackElems == 0, "FP8 tile must be packable by 4");

    #pragma unroll
    for (int pack_idx = lane; pack_idx < kNumPacks; pack_idx += WARP_THREADS) {
        const int elem_idx = pack_idx * kPackElems;
        fp8e4m3_4 packed_in = *reinterpret_cast<fp8e4m3_4*>(vals + elem_idx);
        float4 unpacked = base_types::convertor<float4, fp8e4m3_4>::convert(packed_in);
        unpacked.x *= scale;
        unpacked.y *= scale;
        unpacked.z *= scale;
        unpacked.w *= scale;
        const fp8e4m3_4 packed_out = base_types::convertor<fp8e4m3_4, float4>::convert(unpacked);
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(vals + elem_idx));
        asm volatile("{st.shared.b32 [%0], %1;}" :: "r"(smem_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_out)));
    }
}

template <typename C>
__device__ inline void apply_chunk_scales_to_stage(
    typename globals<C>::A_sc_tile &A_sc_tile,
    typename globals<C>::B_sc_tile (&B_sc_tiles)[C::B_SC_SIZE],
    const float *A_sg_chunks,
    int A_sg_stride,
    const float *A_sg_final,
    int A_sg_final_stride,
    const float *B_sg_chunks,
    int B_sg_stride,
    const float *B_sg_final,
    int B_sg_final_stride,
    int a_chunk_row,
    int b_chunk_row_0,
    int row_block_idx,
    int col_block_idx,
    int chunk_base
) {
    #pragma unroll
    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
        const int chunk_k = chunk_base + ii / 2;
        if (A_sg_chunks != nullptr) {
            const float a_chunk_sg = A_sg_chunks[a_chunk_row * A_sg_stride + chunk_k];
            const float a_final_sg = (A_sg_final != nullptr)
                ? fmaxf(A_sg_final[row_block_idx * A_sg_final_stride], 1.0e-12f)
                : 1.0f;
            const float a_scale = a_chunk_sg / a_final_sg;
            auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&A_sc_tile.data[0]) + 16 * 32 * ii);
            scale_shared_fp8_tile(A_sc_sm_subtile, a_scale);
        }

        if (B_sg_chunks != nullptr) {
            const float b_chunk_sg_0 = B_sg_chunks[b_chunk_row_0 * B_sg_stride + chunk_k];
            const float b_final_sg = (B_sg_final != nullptr)
                ? fmaxf(B_sg_final[col_block_idx * B_sg_final_stride], 1.0e-12f)
                : 1.0f;
            const float b_scale_0 = b_chunk_sg_0 / b_final_sg;
            auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&B_sc_tiles[0].data[0]) + 16 * 32 * ii);
            scale_shared_fp8_tile(B_sc_sm_subtile_0, b_scale_0);

            if constexpr (C::B_SC_SIZE == 2) {
                const float b_chunk_sg_1 = B_sg_chunks[(b_chunk_row_0 + 1) * B_sg_stride + chunk_k];
                const float b_scale_1 = b_chunk_sg_1 / b_final_sg;
                auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                    reinterpret_cast<uint64_t>(&B_sc_tiles[1].data[0]) + 16 * 32 * ii);
                scale_shared_fp8_tile(B_sc_sm_subtile_1, b_scale_1);
            }
        }
    }
}

template <typename C>
__device__ inline void prefetch_residual_tile(
    const globals<C> &g,
    typename globals<C>::D_tile &smem_tile,
    semaphore &residual_load_arrived,
    int row_tile,
    int col_tile
) {
    if constexpr (C::FUSE_RESIDUAL) {
        tma_dev_proxy<typename globals<C>::D_gl> R_proxy(&g.R_tma);
        if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
            tma::expect_bytes(residual_load_arrived, sizeof(typename globals<C>::D_tile));
            tma::load_async(smem_tile, R_proxy, {row_tile, col_tile}, residual_load_arrived);
        }
    }
}

template <typename C>
__device__ inline void add_prefetched_residual_tile(
    typename globals<C>::D_tile &smem_tile,
    rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> &D_reg,
    semaphore &residual_load_arrived,
    uint32_t &residual_load_phase
) {
    if constexpr (C::FUSE_RESIDUAL) {
        wait(residual_load_arrived, residual_load_phase);
        residual_load_phase ^= 1;
        warpgroup::sync(1);

        rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> R_reg;
        warpgroup::load(R_reg, smem_tile);
        warpgroup::sync(1);
        warp::add(D_reg, D_reg, R_reg);
    }
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
    if constexpr (C::FUSE_RESIDUAL) {
        prefetch_residual_tile<C>(
            g, smem_tile, residual_load_arrived, row_tile, col_tile);
        add_prefetched_residual_tile<C>(
            smem_tile, D_reg, residual_load_arrived, residual_load_phase);
    }
}

template <typename C, typename RT>
__device__ inline void write_c1_row_rms_partial(
    const globals<C>& g,
    const RT& D_reg,
    int row_tile,
    int col_tile
) {
    if constexpr (C::FUSE_C1_RMS) {
        rt_fl<RT::rows, RT::cols> D_fl;
        rt_fl<RT::rows, RT::cols> D_sq;
        warp::copy(D_fl, D_reg);
        warp::mul(D_sq, D_fl, D_fl);
        typename decltype(D_sq)::col_vec row_sums;
        warp::row_sum(row_sums, D_sq);

        const int lane = warp::laneid();
        const int warp_row_base =
            row_tile * (C::Mb / 2) + warpgroup::warpid() * RT::rows;
        if ((lane & 3) == 0) {
            #pragma unroll
            for (int i = 0; i < decltype(row_sums)::outer_dim; ++i) {
                const int row_x = warp_row_base + i * 16 + lane / 4;
                const int row_y = row_x + 8;
                g.row_rms_partial[
                    row_x * g.row_rms_partial_stride + col_tile] =
                    row_sums[i][0].x;
                g.row_rms_partial[
                    row_y * g.row_rms_partial_stride + col_tile] =
                    row_sums[i][0].y;
            }
        }
    }
}

template <typename C, typename RT, typename RV>
__device__ inline void accumulate_c1_cta_row_rms_partial(
    const RT& D_reg,
    RV& row_sums,
    bool initialize
) {
    if constexpr (C::FUSE_C1_RMS_CTA) {
        rt_fl<RT::rows, RT::cols> D_fl;
        rt_fl<RT::rows, RT::cols> D_sq;
        warp::copy(D_fl, D_reg);
        warp::mul(D_sq, D_fl, D_fl);
        if (initialize) {
            warp::row_sum(row_sums, D_sq);
        } else {
            warp::row_sum(row_sums, D_sq, row_sums);
        }
    }
}

template <typename C, typename RV>
__device__ inline void write_c1_cta_row_rms_partial(
    const globals<C>& g,
    const RV& row_sums,
    int row_tile,
    int col_block
) {
    if constexpr (C::FUSE_C1_RMS_CTA) {
        const int lane = warp::laneid();
        const int warp_row_base =
            row_tile * (C::Mb / 2) + warpgroup::warpid() * RV::length;
        if ((lane & 3) == 0) {
            #pragma unroll
            for (int i = 0; i < RV::outer_dim; ++i) {
                const int row_x = warp_row_base + i * 16 + lane / 4;
                const int row_y = row_x + 8;
                g.row_rms_partial[
                    row_x * g.row_rms_partial_stride + col_block] =
                    row_sums[i][0].x;
                g.row_rms_partial[
                    row_y * g.row_rms_partial_stride + col_block] =
                    row_sums[i][0].y;
            }
        }
    }
}

template <typename C, bool USE_CHUNK_GRID_SCALE>
__device__ inline void kernel_impl(const globals<C> &g) {
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
        if (g.use_split_D) {
            g.D_K.template prefetch_tma<typename G::D_tile>();
            if (g.v_dim > 0) {
                g.D_V.template prefetch_tma<typename G::D_tile>();
            }
        }
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int N_total = g.use_split_D ? (g.q_dim + g.k_dim + g.v_dim) : g.D.cols();
    const int num_col_blocks = N_total / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles) [C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                      = sm_allocator.allocate<G::outputs_t>();

    // Allocate tensor memory
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_provisioned, tmem_finished;
    __shared__ semaphore tiles_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scale_tiles_ready[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_prepared[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    __shared__ semaphore residual_load_arrived[
        C::PIPELINE_RESIDUAL ? 2 : 1];
    __shared__ float h_reduce[
        C::FUSE_H_NV_CARRIER ? WARPGROUP_WARPS + 1 : 1];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        init_semaphore(tmem_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(scale_tiles_ready[i], 0, 1);
            init_semaphore(scales_prepared[i], 0, C::CLUSTER_SIZE);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, C::CLUSTER_SIZE);
        if constexpr (C::FUSE_RESIDUAL) {
            #pragma unroll
            for (int i = 0; i < (C::PIPELINE_RESIDUAL ? 2 : 1); ++i) {
                init_semaphore(residual_load_arrived[i], 0, 1);
            }
        }
    }
    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait_aligned();

    // Main divergence
    if (warpgroup_id >= C::CONSUMER_WARPGROUPS) {
        // Producer group
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        const int lane = threadIdx.x % WARP_THREADS;
        if (warp_id == 3 && warp::elect_leader()) {
            // Load input tiles to shared memory
            if constexpr (C::USE_PDL) pdl::wait();
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
        } else if (warp_id == 2 && warp::elect_leader()) {
            // Load input scales to shared memory
            if constexpr (C::USE_PDL) pdl::wait();
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
        } else if (warp_id == 1) {
            if constexpr (USE_CHUNK_GRID_SCALE) {
                uint32_t ready_phasebits = 0;
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                    int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                    int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                    for (int i = 0; i < num_red_blocks; ++i) {
                        if (cta_id == 0) {
                            if (lane == 0) {
                                tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                                wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                                arrive(scale_tiles_ready[stage], 1);
                                tma::cluster::arrive(scale_tiles_ready[stage], 1, 1);
                                update_phasebit<0>(phasebits, stage);
                            }
                        } else if (lane == 0) {
                            wait(scale_tiles_ready[stage], get_phasebit<0>(ready_phasebits, stage));
                            update_phasebit<0>(ready_phasebits, stage);
                        }
                        __syncwarp();

                        const int chunk_base = i * (C::Kb / 128);
                        const int a_chunk_row = row_block_idx * 2 + cta_id;
                        const int b_chunk_row_0 = col_block_idx * C::B_SC_SIZE;
                        const int A_sg_stride = g.a_sg_chunk_stride < 0 ? -g.a_sg_chunk_stride : g.a_sg_chunk_stride;
                        const int B_sg_stride = g.b_sg_chunk_stride < 0 ? -g.b_sg_chunk_stride : g.b_sg_chunk_stride;
                        apply_chunk_scales_to_stage<C>(
                            input_scales[stage].A, input_scales[stage].B,
                            g.a_sg_chunk_grid, A_sg_stride,
                            g.a_sg_per_tile, g.a_sg_stride,
                            g.b_sg_chunk_grid, B_sg_stride,
                            g.b_sg_per_tile, g.b_sg_stride,
                            a_chunk_row, b_chunk_row_0,
                            row_block_idx, col_block_idx, chunk_base);

                        __syncwarp();
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        if (lane == 0) {
                            if (cta_id == 0) {
                                arrive(scales_prepared[stage], 1);
                            } else {
                                tma::cluster::arrive(scales_prepared[stage], 0, 1);
                            }
                        }
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            }
        } else if (cta_id == 0 && warp_id == 0 && warp::elect_leader()) {
            // Launch tensor core matrix multiplies
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            uint32_t prepared_phasebits = 0;
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_red_blocks; i++) {
                    if constexpr (USE_CHUNK_GRID_SCALE) {
                        wait(scales_prepared[stage], get_phasebit<0>(prepared_phasebits, stage));
                    } else {
                        tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    }
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
                    if constexpr (USE_CHUNK_GRID_SCALE) {
                        update_phasebit<0>(prepared_phasebits, stage);
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
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // Consumer group
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        const float default_a_sg = g.A_sc_global[{0}];
        const float default_b_sg = g.B_sc_global[{0}];
        uint32_t residual_load_phase[C::PIPELINE_RESIDUAL ? 2 : 1] = {};

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            const bool use_outer_tile_scales =
                (!USE_CHUNK_GRID_SCALE || g.a_sg_chunk_grid != nullptr || g.b_sg_chunk_grid != nullptr);
            const float a_sg = (use_outer_tile_scales && g.a_sg_per_tile != nullptr)
                ? g.a_sg_per_tile[row_block_idx * g.a_sg_stride]
                : default_a_sg;
            const float b_sg = (use_outer_tile_scales && g.b_sg_per_tile != nullptr)
                ? g.b_sg_per_tile[col_block_idx * g.b_sg_stride]
                : default_b_sg;
            const float gs = a_sg * b_sg;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Load the output from tensor memory into registers and store to HBM
            if constexpr (C::FUSE_H_NV_CARRIER) {
                constexpr int SLICE = C::Nb / C::EPI_PIPE_DEPTH;
                constexpr int SLICES_PER_TILE = 128 / SLICE;
                static_assert(SLICE == 32 && C::EPI_PIPE_DEPTH == 8);
                using H_rt_fl = rt_fl<C::Mb / 8, SLICE>;
                using H_rt_bf = rt_bf<C::Mb / 8, SLICE>;
                const int row_tile = row_block_idx * 2 + cta_id;
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
                        warp::mul(f, f, gs);
                        const int col_offset_elems =
                            (C::EPI_PIPE_DEPTH * col_block_idx + i) * SLICE;
                        if (g.silu_dim > 0 &&
                            col_offset_elems < g.silu_dim) {
                            apply_silu_inplace(f);
                        }
                        apply_rope_live64_if_enabled(
                            f, g, row_block_idx, cta_id, col_offset_elems);
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
                            g, output_tiles.D[slot], z[q],
                            residual_load_arrived[0], residual_load_phase[0],
                            row_tile,
                            C::EPI_PIPE_DEPTH * col_block_idx + i);
                        warpgroup::store(output_tiles.D[slot], z[q]);
                        warpgroup::sync(1);
                        warpgroup::tma::store_async<
                            dim::ROW, C::D_CACHE_POLICY>(
                            g.D, output_tiles.D[slot],
                            {row_tile,
                             C::EPI_PIPE_DEPTH * col_block_idx + i});
                    }

                    float sum = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < SLICES_PER_TILE; ++q) {
                        sum += h_nvfp4_tile_carrier::sumsq(z[q]);
                    }
                    const float r = h_nvfp4_tile_carrier::tile_rsqrt(
                        sum, h_reduce, g.h_eps);
                    float amax = 0.0f;
                    #pragma unroll
                    for (int q = 0; q < SLICES_PER_TILE; ++q) {
                        const int i = tile * SLICES_PER_TILE + q;
                        const int col = col_block_idx * C::Nb + i * SLICE;
                        amax = fmaxf(
                            amax,
                            h_nvfp4_tile_carrier::normalized_gamma_amax(
                                z[q], r, g.h_gamma, col));
                    }
                    amax = h_nvfp4_tile_carrier::reduce_max(
                        amax, h_reduce);
                    if (warpgroup::warpid() == 0 &&
                        warp::laneid() == 0) {
                        const int col_tile = col_block_idx * 2 + tile;
                        const int idx =
                            row_tile * (g.h_cols / 128) + col_tile;
                        g.h_r_tile[idx] = r;
                        g.h_amax_tile[idx] = amax;
                    }
                }
            } else if constexpr (C::OVERLAP_EPI) {
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg;
                    warpgroup::load_async(D_reg, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH - 1) {
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0
                    }
                    warp::mul(D_reg, D_reg, gs);
                    int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                    // SiLU epilogue: apply to tiles in [0, silu_dim) columns
                    if (g.silu_dim > 0) {
                        if (col_offset_elems < g.silu_dim) {
                            apply_silu_inplace(D_reg);
                        }
                    }
                    apply_rope_live64_if_enabled(D_reg, g, row_block_idx, cta_id, col_offset_elems);
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    if constexpr (C::FUSE_RESIDUAL) {
                        rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_bf;
                        warp::copy(D_reg_bf, D_reg);
                        maybe_add_residual_tile<C>(
                            g, output_tiles.D[i%C::NUM_D_TILES], D_reg_bf,
                            residual_load_arrived[0], residual_load_phase[0],
                            row_block_idx * 2 + cta_id,
                            C::EPI_PIPE_DEPTH * col_block_idx + i);
                        write_c1_row_rms_partial<C>(
                            g, D_reg_bf,
                            row_block_idx * 2 + cta_id,
                            C::EPI_PIPE_DEPTH * col_block_idx + i);
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg_bf);
                    } else {
                        warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg);
                    }
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
                using c1_slice_t =
                    rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH>;
                typename c1_slice_t::col_vec c1_cta_row_sums;
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    warp::mul(D_reg_fl, D_reg_fl, gs);
                    int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                    // SiLU epilogue: apply to tiles in [0, silu_dim) columns
                    if (g.silu_dim > 0) {
                        if (col_offset_elems < g.silu_dim) {
                            apply_silu_inplace(D_reg_fl);
                        }
                    }
                    apply_rope_live64_if_enabled(D_reg_fl, g, row_block_idx, cta_id, col_offset_elems);
                    warp::copy(D_reg[i], D_reg_fl);
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1); // signal CTA 0
                if constexpr (C::PIPELINE_RESIDUAL) {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        prefetch_residual_tile<C>(
                            g, output_tiles.R[i], residual_load_arrived[i],
                            row_block_idx * 2 + cta_id,
                            C::EPI_PIPE_DEPTH * col_block_idx + i);
                    }
                }
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    if constexpr (C::PIPELINE_RESIDUAL) {
                        constexpr int residual_pipe_depth = 2;
                        const int slot = i % residual_pipe_depth;
                        add_prefetched_residual_tile<C>(
                            output_tiles.R[slot], D_reg[i],
                            residual_load_arrived[slot],
                            residual_load_phase[slot]);
                        if (i + residual_pipe_depth < C::EPI_PIPE_DEPTH) {
                            prefetch_residual_tile<C>(
                                g, output_tiles.R[slot],
                                residual_load_arrived[slot],
                                row_block_idx * 2 + cta_id,
                                C::EPI_PIPE_DEPTH * col_block_idx +
                                    i + residual_pipe_depth);
                        }
                    } else {
                        maybe_add_residual_tile<C>(
                            g, output_tiles.D[i%C::NUM_D_TILES], D_reg[i],
                            residual_load_arrived[0], residual_load_phase[0],
                            row_block_idx * 2 + cta_id,
                            C::EPI_PIPE_DEPTH * col_block_idx + i);
                    }
                    if constexpr (!C::FUSE_C1_RMS_CTA) {
                        write_c1_row_rms_partial<C>(
                            g, D_reg[i],
                            row_block_idx * 2 + cta_id,
                            C::EPI_PIPE_DEPTH * col_block_idx + i);
                    }
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
                    if constexpr (C::FUSE_C1_RMS_CTA) {
                        accumulate_c1_cta_row_rms_partial<C>(
                            D_reg[i], c1_cta_row_sums, i == 0);
                    }
                }
                write_c1_cta_row_rms_partial<C>(
                    g, c1_cta_row_sums,
                    row_block_idx * 2 + cta_id, col_block_idx);
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        // Ensure all TMA stores have committed before signaling the next
        // kernel can start. Without this wait, pdl::arrive() can fire while
        // async TMA stores are still in-flight, causing the next kernel to
        // read stale output data (manifests as cudaErrorLaunchFailure at
        // large M when kernels are launched back-to-back).
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) {
            if (warp::elect_leader()) tma::cluster::arrive(tmem_finished, 1-cta_id);
            wait(tmem_finished, 0);
            tm_allocator.deprovision();
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    kernel_impl<C, false>(g);
}

template <typename C>
__device__ inline void kernel_virtual_rescale(const globals<C> &g) {
    kernel_impl<C, true>(g);
}

template <typename C>
__device__ inline void kernel_chunk_grid(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::Kb == 128, "chunk-grid v4 consumer applies one fp32 SG per 128-wide reduction chunk");

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
        if (g.use_split_D) {
            g.D_K.template prefetch_tma<typename G::D_tile>();
            if (g.v_dim > 0) {
                g.D_V.template prefetch_tma<typename G::D_tile>();
            }
        }
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int N_total = g.use_split_D ? (g.q_dim + g.k_dim + g.v_dim) : g.D.cols();
    const int num_col_blocks = N_total / C::Nb;
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
    static constexpr int OUTPUT_PIPE_DEPTH = C::Nb == 128 ? 2 : 1;
    __shared__ semaphore outputs_arrived[OUTPUT_PIPE_DEPTH];
    __shared__ semaphore outputs_finished[OUTPUT_PIPE_DEPTH];
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arrived[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        #pragma unroll
        for (int i = 0; i < OUTPUT_PIPE_DEPTH; ++i) {
            init_semaphore(outputs_arrived[i], 0, 1);
            init_semaphore(outputs_finished[i], 0, C::CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();
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
            if constexpr (C::USE_PDL) pdl::wait();
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
                    if constexpr (C::B_SC_SIZE == 2) {
                        tma::cluster::load_async(input_scales[stage].B[cta_id], g.B_sc, {col_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    } else if (cta_id == 0) {
                        tma::cluster::load_async(input_scales[stage].B[0], g.B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            uint32_t output_phasebits = 0xFFFF0000;

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                for (int i = 0; i < num_red_blocks; ++i) {
                    const int output_slot = OUTPUT_PIPE_DEPTH == 2 ? (i & 1) : 0;
                    auto &out_tm = output_slot == 0 ? out_tm_0 : out_tm_1;
                    wait(outputs_finished[output_slot], get_phasebit<1>(output_phasebits, output_slot));
                    tensor_after_thread_sync();

                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
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
                    mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                            A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                            B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                            inputs_finished[stage]);
                    tensor_commit<2>(outputs_arrived[output_slot]);
                    update_phasebit<1>(output_phasebits, output_slot);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
        uint32_t output_phasebits = 0xFFFF0000;
        const int A_sg_stride = g.a_sg_stride < 0 ? -g.a_sg_stride : g.a_sg_stride;
        const int B_sg_stride = g.b_sg_stride < 0 ? -g.b_sg_stride : g.b_sg_stride;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_acc[C::EPI_PIPE_DEPTH];

            {
                constexpr int i = 0;
                const int output_slot = 0;
                auto &out_tm = out_tm_0;
                wait(outputs_arrived[output_slot], get_phasebit<0>(output_phasebits, output_slot));

                const int a_chunk_row = row_block_idx * C::CLUSTER_SIZE + cta_id;
                const float a_sg = g.a_sg_per_tile[a_chunk_row * A_sg_stride + i];

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::load_async(
                        D_acc[epi],
                        out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*epi));
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished[output_slot], 0, 1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    constexpr int cols_per_epi = C::Nb / C::EPI_PIPE_DEPTH;
                    const int b_chunk_col = col_block_idx * (C::Nb / 128) + (epi * cols_per_epi) / 128;
                    const float b_sg = g.b_sg_per_tile[b_chunk_col * B_sg_stride + i];
                    const float gs = a_sg * b_sg;
                    warp::mul(D_acc[epi], D_acc[epi], gs);
                }
                warpgroup::sync(1);
                tensor_after_thread_sync();
                update_phasebit<0>(output_phasebits, output_slot);
            }

            for (int i = 1; i < num_red_blocks; ++i) {
                const int output_slot = OUTPUT_PIPE_DEPTH == 2 ? (i & 1) : 0;
                auto &out_tm = output_slot == 0 ? out_tm_0 : out_tm_1;
                wait(outputs_arrived[output_slot], get_phasebit<0>(output_phasebits, output_slot));

                const int a_chunk_row = row_block_idx * C::CLUSTER_SIZE + cta_id;
                const float a_sg = g.a_sg_per_tile[a_chunk_row * A_sg_stride + i];

                rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_pipe[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::load_async(
                        D_pipe[epi],
                        out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*epi));
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished[output_slot], 0, 1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    constexpr int cols_per_epi = C::Nb / C::EPI_PIPE_DEPTH;
                    const int b_chunk_col = col_block_idx * (C::Nb / 128) + (epi * cols_per_epi) / 128;
                    const float b_sg = g.b_sg_per_tile[b_chunk_col * B_sg_stride + i];
                    const float gs = a_sg * b_sg;
                    add_scaled_inplace(D_acc[epi], D_pipe[epi], gs);
                }
                warpgroup::sync(1);
                tensor_after_thread_sync();
                update_phasebit<0>(output_phasebits, output_slot);
            }

            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                if (g.silu_dim > 0) {
                    if (col_offset_elems < g.silu_dim) {
                        apply_silu_inplace(D_acc[i]);
                    }
                }
                apply_rope_live64_if_enabled(D_acc[i], g, row_block_idx, cta_id, col_offset_elems);
                rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg;
                warp::copy(D_reg, D_acc[i]);
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
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

} // namespace nvfp4_gemm
