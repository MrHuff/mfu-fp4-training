#pragma once
// ================================================================
// MXFP4 Split3 One-Pass Accum GEMM
// Specialized for 3-way concatenated dgrad:
//   D = A0 x B0^T + A1 x B1^T + A2 x B2^T
// where A_i are strided slices of one full MXFP4 row buffer.
// ================================================================

#include "mxfp4_gemm.cuh"

namespace mxfp4_split3_accum_gemm {

using namespace kittens;

static constexpr int NUM_SPLITS = 3;

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE,
          int _NUM_D_TILES, bool _OVERLAP_EPI, int _CLUSTER_SIZE = 2, bool _USE_PDL = false>
struct config {
    static_assert(_Nb == 128 || _Nb == 256, "Nb must be 128 or 256");
    static_assert(_LOAD_PIPE_DEPTH > 0 && _LOAD_PIPE_DEPTH <= 5, "LOAD_PIPE_DEPTH must be > 0 and <= 5");
    static_assert(_EPI_PIPE_DEPTH > 0, "EPI_PIPE_DEPTH must be > 0");
    static_assert(_SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be > 0");
    static_assert(_NUM_D_TILES > 0, "NUM_D_TILES must be > 0");
    static_assert(_EPI_PIPE_DEPTH <= 1 || _NUM_D_TILES >= 2,
                  "NUM_D_TILES must be at least 2 if EPI_PIPE_DEPTH > 1");

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
    static constexpr int Mb = 256;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 128;
    static constexpr int NUM_D_TILES = _NUM_D_TILES;
};

template <typename _GL>
struct tma_dev_proxy {
    using identifier = ducks::gl::identifier;
    using T = typename _GL::T;
    using T2 = typename _GL::T2;
    using dtype = typename _GL::dtype;
    static constexpr int __b__ = _GL::__b__, __d__ = _GL::__d__, __r__ = _GL::__r__, __c__ = _GL::__c__;

    const CUtensorMap *dev_tma;

    __device__ explicit tma_dev_proxy(const CUtensorMap *_dev_tma) : dev_tma(_dev_tma) {}

    template<int axis> __device__ inline size_t shape() const { return 0; }
    template<int axis> __device__ inline size_t stride() const { return 0; }

    template<typename U, int axis> __device__ inline const CUtensorMap *get_tma() const {
        return dev_tma;
    }
    __device__ inline void prefetch() const {
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(dev_tma)) : "memory");
    }
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb / 2, C::Kb / 2>;
    using A_sc_tile = st_fp8e8m0<32, 16, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb / 2, C::Kb / 2>;
    using B_sc_tile = st_fp8e8m0<32, 16, false>;
    using D_tile = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl = gl<fp8e8m0, -1, -1, 32, 16, B_sc_tile>;
    using D_gl = gl<bf16, 1, 1, -1, -1, D_tile>;

    CUtensorMap A_tma[NUM_SPLITS];
    CUtensorMap A_sc_tma[NUM_SPLITS];
    CUtensorMap B_tma[NUM_SPLITS];
    CUtensorMap B_sc_tma[NUM_SPLITS];
    CUtensorMap D_tma;

    int num_red_blocks[NUM_SPLITS];
    int num_row_blocks;
    int num_col_blocks;

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
        int spatial_tiles = num_row_blocks * num_col_blocks;
        int grid_size = min(spatial_tiles, num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int shm = sizeof(input_tiles_t) * C::LOAD_PIPE_DEPTH + 1024 +
                            sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                            sizeof(outputs_t);
        static_assert(shm <= MAX_SHARED_MEMORY - 1024);
        return shm;
    }
};

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    const int num_blocks = g.num_row_blocks * g.num_col_blocks;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int split = 0; split < NUM_SPLITS; ++split) {
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[split])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[split])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[split])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[split])) : "memory");
        }
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.D_tma)) : "memory");
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * g.num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    typename G::input_tiles_t (&input_tiles)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] =
        sm_allocator.allocate<G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t &output_tiles = sm_allocator.allocate<G::outputs_t>();

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

    tma_dev_proxy<typename G::D_gl> proxy_D(&g.D_tma);

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS * C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                #pragma unroll
                for (int split = 0; split < NUM_SPLITS; ++split) {
                    tma_dev_proxy<typename G::A_fp4x2_gl> proxy_A(&g.A_tma[split]);
                    tma_dev_proxy<typename G::B_fp4x2_gl> proxy_B(&g.B_tma[split]);
                    for (int i = 0; i < g.num_red_blocks[split]; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        tma::cluster::load_async(input_tiles[stage].A, proxy_A, {row_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        tma::cluster::load_async(input_tiles[stage].B, proxy_B, {col_block_idx * 2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                #pragma unroll
                for (int split = 0; split < NUM_SPLITS; ++split) {
                    tma_dev_proxy<typename G::A_sc_gl> proxy_A_sc(&g.A_sc_tma[split]);
                    tma_dev_proxy<typename G::B_sc_gl> proxy_B_sc(&g.B_sc_tma[split]);
                    for (int i = 0; i < g.num_red_blocks[split]; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                            tma::cluster::load_async(
                                input_scales[stage].A[k],
                                proxy_A_sc,
                                {row_block_idx * 2 + cta_id, i * C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(1 << cta_id),
                                0);
                        }
                        if constexpr (C::B_SC_SIZE == 2) {
                            #pragma unroll
                            for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                                tma::cluster::load_async(
                                    input_scales[stage].B[cta_id * C::MMA_PER_TILE + k],
                                    proxy_B_sc,
                                    {col_block_idx * 2 + cta_id, i * C::MMA_PER_TILE + k, 0, 0},
                                    scales_arrived[stage],
                                    (uint16_t)(0b11),
                                    0);
                            }
                        } else if (cta_id == 0) {
                            #pragma unroll
                            for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                                tma::cluster::load_async(
                                    input_scales[stage].B[k],
                                    proxy_B_sc,
                                    {col_block_idx, i * C::MMA_PER_TILE + k, 0, 0},
                                    scales_arrived[stage],
                                    (uint16_t)(0b11),
                                    0);
                            }
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                bool first_mma = true;

                #pragma unroll
                for (int split = 0; split < NUM_SPLITS; ++split) {
                    for (int i = 0; i < g.num_red_blocks[split]; ++i) {
                        tma::expect_bytes(scales_arrived[stage], 2 * sizeof(typename G::input_scales_t));
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
                        tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(typename G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        if (first_mma) {
                            mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                    A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                    B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                    inputs_finished[stage]);
                            first_mma = false;
                        } else {
                            mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                     A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                     B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                     inputs_finished[stage]);
                        }
                        update_phasebit<0>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }

                tensor_commit<2>(outputs_arrived);
                update_phasebit<1>(phasebits, 0);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        if constexpr (C::USE_PDL) warpgroup::pdl::wait();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
            rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                warpgroup::load_async(
                    D_reg_fl,
                    out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                        0,
                        C::Nb / C::EPI_PIPE_DEPTH * i
                    )
                );
                warp::mul(D_reg_fl, D_reg_fl, MXFP4_ALPHA);
                warp::copy(D_reg[i], D_reg_fl);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i % C::NUM_D_TILES], D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    proxy_D, output_tiles.D[i % C::NUM_D_TILES],
                    {row_block_idx * 2 + cta_id, C::EPI_PIPE_DEPTH * col_block_idx + i});
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

} // namespace mxfp4_split3_accum_gemm
