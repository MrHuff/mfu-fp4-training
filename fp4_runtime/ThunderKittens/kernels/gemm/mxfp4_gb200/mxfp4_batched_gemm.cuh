#pragma once
// ================================================================
// MXFP4 True Batched GEMM Kernel
// D_i = A_i x B_i^T, independently per batch.
// No global scale — E8M0 scales are self-contained.
// Requires mxfp4_gemm.cuh for config struct.
// ================================================================

#include "mxfp4_gemm.cuh"

// ================================================================
// True Batched GEMM: B independent GEMMs, D_i = A_i × B_i^T
//
// Grid: (spatial_tiles, 1, num_batches)
// Each CTA handles exactly one batch, indexed by blockIdx.z.
// Per-batch TMA descriptors are created once per CTA.
// ================================================================
namespace mxfp4_batched_gemm {

static constexpr int MAX_BATCHES = 32;

// ---- tma_dev_proxy: gl-like wrapper that returns a device-global CUtensorMap* ----
template <typename _GL>
struct tma_dev_proxy {
    using identifier = ducks::gl::identifier;
    using T     = typename _GL::T;
    using T2    = typename _GL::T2;
    using dtype = typename _GL::dtype;
    static constexpr int __b__ = _GL::__b__, __d__ = _GL::__d__, __r__ = _GL::__r__, __c__ = _GL::__c__;

    const CUtensorMap* dev_tma;

    __device__ tma_dev_proxy(const CUtensorMap* _dev_tma) : dev_tma(_dev_tma) {}

    template<int axis> __device__ inline size_t shape() const { return 0; }
    template<int axis> __device__ inline size_t stride() const { return 0; }

    template<typename U, int axis> __device__ inline const CUtensorMap* get_tma() const {
        return dev_tma;
    }
    __device__ inline void prefetch() const {
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(dev_tma)) : "memory");
    }
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<fp8e8m0,    -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<fp8e8m0,    -1, -1, 32, 16, B_sc_tile>;
    using D_gl           = gl<bf16,        1,  1, -1, -1, D_tile>;

    CUtensorMap A_tma[MAX_BATCHES];
    CUtensorMap A_sc_tma[MAX_BATCHES];
    CUtensorMap B_tma[MAX_BATCHES];
    CUtensorMap B_sc_tma[MAX_BATCHES];
    CUtensorMap D_tma[MAX_BATCHES];
    mxfp4_rope_epilogue::rope_desc rope[MAX_BATCHES];
    mxfp4_rope_epilogue::rope_live64_desc rope_live64[MAX_BATCHES];

    int       num_red_blocks;
    int       num_batches;
    int       num_row_blocks;
    int       num_col_blocks;
    int       num_red_blocks_by_batch[MAX_BATCHES];
    int       num_row_blocks_by_batch[MAX_BATCHES];
    int       num_col_blocks_by_batch[MAX_BATCHES];
    int       tile_offsets[MAX_BATCHES + 1];
    int       total_spatial_tiles;
    bool      uniform_strided;
    int       a_row_block_stride;
    int       a_k_block_stride;
    int       b_row_block_stride;
    int       b_k_block_stride;
    int       d_row_block_stride;
    int       a_k_block_offset;
    int       b_k_block_offset;
    const uint8_t* tilemask_ptr;
    int       tilemask_rows;
    int       tilemask_cols;
    bool      tilemask_transposed;
    bool      output_causal;

    struct input_tiles_t {
        A_fp4x2_tile A;
        B_fp4x2_tile B;
    };
    struct input_scales_t {
        A_sc_tile A[C::MMA_PER_TILE];
        B_sc_tile B[C::B_SC_SIZE * C::MMA_PER_TILE];
    };
    struct outputs_t {
        D_tile D[C::NUM_D_TILES];
    };

    __host__ inline dim3 grid() const {
        int spatial_tiles = total_spatial_tiles > 0
            ? total_spatial_tiles
            : num_row_blocks * num_col_blocks;
        int x = min(spatial_tiles, num_sms());
        if constexpr (C::CLUSTER_SIZE > 1) {
            int rem = x % C::CLUSTER_SIZE;
            if (rem != 0) {
                x += C::CLUSTER_SIZE - rem;
            }
        }
        int z = total_spatial_tiles > 0 ? 1 : num_batches;
        return dim3(x, 1, z);
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
__device__ inline bool reduction_iter_active(
    const globals<C> &g,
    int row_tile_128,
    int red_iter
) {
    if (g.tilemask_ptr == nullptr) {
        return true;
    }

    constexpr int RED_TILES_PER_ITER = C::Kb / 128;
    const int red_tile_base = red_iter * RED_TILES_PER_ITER;

    auto tile_active = [&](int red_tile_128) {
        if (g.tilemask_transposed) {
            if (red_tile_128 >= g.tilemask_rows || row_tile_128 >= g.tilemask_cols) {
                return false;
            }
            return g.tilemask_ptr[red_tile_128 * g.tilemask_cols + row_tile_128] != 0;
        }
        if (row_tile_128 >= g.tilemask_rows || red_tile_128 >= g.tilemask_cols) {
            return false;
        }
        return g.tilemask_ptr[row_tile_128 * g.tilemask_cols + red_tile_128] != 0;
    };

    bool active = false;
    #pragma unroll
    for (int red_tile_offset = 0; red_tile_offset < RED_TILES_PER_ITER; ++red_tile_offset) {
        active = active || tile_active(red_tile_base + red_tile_offset);
    }
    return active;
}

template <typename C, bool ATBT>
__device__ inline bool reduction_iter_active_for_output_tile(
    const globals<C> &g,
    int row_block_idx,
    int col_block_idx,
    int red_iter
) {
    if (g.tilemask_ptr == nullptr) {
        return true;
    }
    const int mask_block_idx = ATBT ? col_block_idx : row_block_idx;
    bool active = false;
    #pragma unroll
    for (int tile = 0; tile < C::CLUSTER_SIZE; ++tile) {
        active = active || reduction_iter_active(g, mask_block_idx * C::CLUSTER_SIZE + tile, red_iter);
    }
    return active;
}

template <typename C>
__device__ inline bool output_block_active(
    const globals<C> &g,
    int row_block_idx,
    int col_block_idx
) {
    if (!g.output_causal) {
        return true;
    }
    return col_block_idx <= row_block_idx;
}

template <typename C>
__device__ inline void resolve_problem_tile(
    const globals<C> &g,
    int flat_block_idx,
    int legacy_batch,
    int &batch,
    int &block_idx,
    int &num_row_blocks,
    int &num_col_blocks,
    int &num_red_blocks
) {
    if (g.total_spatial_tiles > 0) {
        int selected = 0;
        #pragma unroll
        for (int i = 0; i < MAX_BATCHES; ++i) {
            if (i < g.num_batches && flat_block_idx >= g.tile_offsets[i]) {
                selected = i;
            }
        }
        batch = selected;
        block_idx = flat_block_idx - g.tile_offsets[selected];
        num_row_blocks = g.num_row_blocks_by_batch[selected];
        num_col_blocks = g.num_col_blocks_by_batch[selected];
        num_red_blocks = g.num_red_blocks_by_batch[selected];
    } else {
        batch = legacy_batch;
        block_idx = flat_block_idx;
        num_row_blocks = g.num_row_blocks;
        num_col_blocks = g.num_col_blocks;
        num_red_blocks = g.num_red_blocks;
    }
}

template <typename C>
__device__ inline void resolve_block_coords(
    int block_idx,
    int num_row_blocks,
    int num_col_blocks,
    int &row_block_idx,
    int &col_block_idx
) {
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    int supergroup_idx = block_idx / num_blocks_per_supergroup;
    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
    int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
    int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
    row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
    col_block_idx = idx_within_supergroup / rows_in_supergroup;
}

template <typename C, bool ATBT = false, bool APPLY_ROPE = true>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    const int legacy_batch = blockIdx.z;
    const int num_blocks = g.total_spatial_tiles > 0 ? g.total_spatial_tiles : g.num_row_blocks * g.num_col_blocks;

    // Prefetch this batch's TMA descriptors
    if (threadIdx.x == 0) {
        if (g.uniform_strided) {
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[0])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[0])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[0])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[0])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.D_tma[0])) : "memory");
        } else if (g.total_spatial_tiles > 0) {
            for (int b = 0; b < g.num_batches; ++b) {
                asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[b])) : "memory");
                asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[b])) : "memory");
                asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[b])) : "memory");
                asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[b])) : "memory");
                asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.D_tma[b])) : "memory");
            }
        } else {
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[legacy_batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[legacy_batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[legacy_batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[legacy_batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.D_tma[legacy_batch])) : "memory");
        }
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int cluster_stride = max(1, gridDim.x / C::CLUSTER_SIZE);
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
    everyone::tma::cluster::wait_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // ── Producer: load input tiles ──
            if constexpr (C::USE_PDL) pdl::wait();

            for (int flat_block_idx = cluster_id; flat_block_idx < num_blocks; flat_block_idx += cluster_stride) {
                int batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks;
                int row_block_idx, col_block_idx;
                resolve_problem_tile<C>(g, flat_block_idx, legacy_batch, batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks);
                resolve_block_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);
                if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                    continue;
                }
                const int tma_batch = g.uniform_strided ? 0 : batch;
                const int a_row_block_base = g.uniform_strided ? batch * g.a_row_block_stride : 0;
                const int a_k_block_base = (g.uniform_strided ? batch * g.a_k_block_stride : 0) + g.a_k_block_offset;
                const int b_row_block_base = g.uniform_strided ? batch * g.b_row_block_stride : 0;
                const int b_k_block_base = (g.uniform_strided ? batch * g.b_k_block_stride : 0) + g.b_k_block_offset;
                const int row_tile_128_0 = row_block_idx * 2 + 0;
                const int row_tile_128_1 = row_block_idx * 2 + 1;
                tma_dev_proxy<typename G::A_fp4x2_gl> proxy_A(&g.A_tma[tma_batch]);
                tma_dev_proxy<typename G::B_fp4x2_gl> proxy_B(&g.B_tma[tma_batch]);

                for (int i = 0; i < num_red_blocks; ++i) {
                    const bool block_iter_active =
                        reduction_iter_active_for_output_tile<C, ATBT>(g, row_block_idx, col_block_idx, i);
                    if (!block_iter_active) continue;
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if constexpr (ATBT) {
                        tma::cluster::load_async(input_tiles[stage].A, proxy_A, {a_row_block_base + i * C::CLUSTER_SIZE + cta_id, a_k_block_base + row_block_idx}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    } else {
                        tma::cluster::load_async(input_tiles[stage].A, proxy_A, {a_row_block_base + row_block_idx*2 + cta_id, a_k_block_base + i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    }
                    tma::cluster::load_async(input_tiles[stage].B, proxy_B, {b_row_block_base + col_block_idx*2 + cta_id, b_k_block_base + i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            // ── Producer: load input scales ──
            if constexpr (C::USE_PDL) pdl::wait();

            for (int flat_block_idx = cluster_id; flat_block_idx < num_blocks; flat_block_idx += cluster_stride) {
                int batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks;
                int row_block_idx, col_block_idx;
                resolve_problem_tile<C>(g, flat_block_idx, legacy_batch, batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks);
                resolve_block_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);
                if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                    continue;
                }
                const int tma_batch = g.uniform_strided ? 0 : batch;
                const int a_row_block_base = g.uniform_strided ? batch * g.a_row_block_stride : 0;
                const int a_sc_col_block_base = (g.uniform_strided ? batch * g.a_k_block_stride : 0) + g.a_k_block_offset;
                const int a_sc_k_block_base = (g.uniform_strided ? batch * g.a_k_block_stride : 0) * C::MMA_PER_TILE + g.a_k_block_offset * C::MMA_PER_TILE;
                const int b_sc_row_block_base = g.uniform_strided ? batch * g.b_row_block_stride : 0;
                const int b_sc_k_block_base = (g.uniform_strided ? batch * g.b_k_block_stride : 0) * C::MMA_PER_TILE + g.b_k_block_offset * C::MMA_PER_TILE;
                const int row_tile_128_0 = row_block_idx * 2 + 0;
                const int row_tile_128_1 = row_block_idx * 2 + 1;
                tma_dev_proxy<typename G::A_sc_gl> proxy_A_sc(&g.A_sc_tma[tma_batch]);
                tma_dev_proxy<typename G::B_sc_gl> proxy_B_sc(&g.B_sc_tma[tma_batch]);

                for (int i = 0; i < num_red_blocks; ++i) {
                    const bool block_iter_active =
                        reduction_iter_active_for_output_tile<C, ATBT>(g, row_block_idx, col_block_idx, i);
                    if (!block_iter_active) continue;
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                        if constexpr (ATBT) {
                            tma::cluster::load_async(
                                input_scales[stage].A[k],
                                proxy_A_sc,
                                {a_row_block_base + i * C::MMA_PER_TILE + k, a_sc_col_block_base + row_block_idx * C::CLUSTER_SIZE + cta_id, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(1<<cta_id),
                                0
                            );
                        } else {
                            tma::cluster::load_async(
                                input_scales[stage].A[k],
                                proxy_A_sc,
                                {a_row_block_base + row_block_idx*2 + cta_id, a_sc_k_block_base + i * C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(1<<cta_id),
                                0
                            );
                        }
                    }
                    if constexpr (C::B_SC_SIZE == 2) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                            tma::cluster::load_async(
                                input_scales[stage].B[cta_id * C::MMA_PER_TILE + k],
                                proxy_B_sc,
                                {b_sc_row_block_base + col_block_idx*2 + cta_id, b_sc_k_block_base + i * C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(0b11),
                                0
                            );
                        }
                    } else if (cta_id == 0) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                            tma::cluster::load_async(
                            input_scales[stage].B[k],
                            proxy_B_sc,
                            {b_sc_row_block_base + col_block_idx, b_sc_k_block_base + i * C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(0b11),
                                0
                            );
                        }
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // ── MMA warp ──
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

            for (int flat_block_idx = cluster_id; flat_block_idx < num_blocks; flat_block_idx += cluster_stride) {
                int batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks;
                int row_block_idx, col_block_idx;
                resolve_problem_tile<C>(g, flat_block_idx, legacy_batch, batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks);
                resolve_block_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);
                if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                    continue;
                }
                const int row_tile_128_0 = row_block_idx * 2 + 0;
                const int row_tile_128_1 = row_block_idx * 2 + 1;
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                bool issued_mma = false;
                for (int i = 0; i < num_red_blocks; i++) {
                    const bool block_iter_active =
                        reduction_iter_active_for_output_tile<C, ATBT>(g, row_block_idx, col_block_idx, i);
                    if (!block_iter_active) continue;
                    tma::expect_bytes(scales_arrived[stage], 2*sizeof(G::input_scales_t));
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                        auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * C::MMA_PER_TILE * 16 + k * 16);
                        load_mxnv_scale_async2(A_sc_tm_subtile, input_scales[stage].A[k]);
                        auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * C::MMA_PER_TILE * 32 + k * C::B_SC_SIZE * 16);
                        load_mxnv_scale_async2(B_sc_tm_subtile_0, input_scales[stage].B[k]);
                        if constexpr (C::B_SC_SIZE == 2) {
                            auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * C::MMA_PER_TILE * 32 + k * C::B_SC_SIZE * 16 + 16);
                            load_mxnv_scale_async2(B_sc_tm_subtile_1, input_scales[stage].B[C::MMA_PER_TILE + k]);
                        }
                    }
                    tma::expect_bytes(tiles_arrived[stage], 2*sizeof(G::input_tiles_t));
                    wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                    auto A_sc_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_subtile = B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32);
                    if (!issued_mma) {
                        if constexpr (ATBT) {
                            kittens::mma<transpose::T, transpose::T, decltype(out_tm), typename G::A_fp4x2_tile, typename G::B_fp4x2_tile, decltype(A_sc_subtile), decltype(B_sc_subtile), 0, C::CLUSTER_SIZE>(
                                out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_subtile, B_sc_subtile, inputs_finished[stage]);
                        } else {
                            mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                    A_sc_subtile,
                                    B_sc_subtile,
                                    inputs_finished[stage]);
                        }
                        issued_mma = true;
                    } else {
                        if constexpr (ATBT) {
                            kittens::mma<transpose::T, transpose::T, decltype(out_tm), typename G::A_fp4x2_tile, typename G::B_fp4x2_tile, decltype(A_sc_subtile), decltype(B_sc_subtile), 1, C::CLUSTER_SIZE>(
                                out_tm, input_tiles[stage].A, input_tiles[stage].B, A_sc_subtile, B_sc_subtile, inputs_finished[stage]);
                        } else {
                            mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                     A_sc_subtile,
                                     B_sc_subtile,
                                     inputs_finished[stage]);
                        }
                    }
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        // ── Consumer: direct store (no global scale for MXFP4) ──
        if constexpr (C::USE_PDL) warpgroup::pdl::wait();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        for (int flat_block_idx = cluster_id; flat_block_idx < num_blocks; flat_block_idx += cluster_stride) {
            int batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks;
            int row_block_idx, col_block_idx;
            resolve_problem_tile<C>(g, flat_block_idx, legacy_batch, batch, block_idx, num_row_blocks, num_col_blocks, num_red_blocks);
            resolve_block_coords<C>(block_idx, num_row_blocks, num_col_blocks, row_block_idx, col_block_idx);
            if (!output_block_active<C>(g, row_block_idx, col_block_idx)) {
                continue;
            }
            const int tma_batch = g.uniform_strided ? 0 : batch;
            const int d_row_block_base = g.uniform_strided ? batch * g.d_row_block_stride : 0;
            const int rope_batch = g.uniform_strided ? 0 : batch;
            tma_dev_proxy<typename G::D_gl> proxy_D(&g.D_tma[tma_batch]);

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Non-overlapping epilogue: load all subtiles, apply the same
            // MXFP4 dequant correction as the single-GEMM kernel, then store.
            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
            rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                warpgroup::load_async(D_reg_fl[i], out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                warp::mul(D_reg_fl[i], D_reg_fl[i], MXFP4_ALPHA);
                if constexpr (APPLY_ROPE) {
                    if (g.rope_live64[rope_batch].enabled()) {
                        if constexpr (C::ROPE_LIVE64_RHT32) {
                            mxfp4_rope_epilogue::apply_inplace_live64_rht32(
                                D_reg_fl[i],
                                g.rope_live64[rope_batch],
                                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                                (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                            );
                        } else {
                            mxfp4_rope_epilogue::apply_inplace_live64(
                                D_reg_fl[i],
                                g.rope_live64[rope_batch],
                                (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                                (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                            );
                        }
                    } else {
                        mxfp4_rope_epilogue::apply_inplace(
                            D_reg_fl[i],
                            g.rope[rope_batch],
                            (row_block_idx * 2 + cta_id) * (C::Mb / 2),
                            (col_block_idx * C::EPI_PIPE_DEPTH + i) * (C::Nb / C::EPI_PIPE_DEPTH)
                        );
                    }
                }
            }
            rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++)
                warp::copy(D_reg[i], D_reg_fl[i]);
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(proxy_D, output_tiles.D[i%C::NUM_D_TILES], {d_row_block_base + row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
        warpgroup::sync(1);
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
    }

    // Keep both clustered CTAs resident until peer DSM/TMEM tail traffic retires.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

} // namespace mxfp4_batched_gemm
