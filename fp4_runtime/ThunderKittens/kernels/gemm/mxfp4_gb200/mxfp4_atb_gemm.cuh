#pragma once
// ================================================================
// MXFP4 AtB GEMM prototype.
//
// Computes D = A^T x B for row-major packed FP4 operands:
//   A: K x (M/2), scales (M/128) x (K/128) x 32 x 16
//   B: K x (N/2), scales (N/128) x (K/128) x 32 x 16
//   D: M x N BF16
//
// The payloads are row-major in the unreduced K dimension. The scale tensors
// are intentionally in logical MMA orientation, matching the output M/N axes
// first, because tcgen05 microscale descriptors currently assume ABt-style
// scale offsets even when the payload descriptor is transposed.
//
// This is intentionally narrow: it is for the FA backward dK path where
// row-major dS can feed dS^T x Q without materializing column-major dS.
// ================================================================

#include "mxfp4_gemm.cuh"

namespace mxfp4_atb_gemm {

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb / 2, C::Kb / 2>;
    using A_sc_tile    = st_fp8e8m0<32, 16, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb / 2, C::Kb / 2>;
    using B_sc_tile    = st_fp8e8m0<32, 16, false>;
    using D_tile       = st_bf<C::Mb / 2, C::Nb / C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2, 1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<fp8e8m0,  -1, -1, 32, 16, B_sc_tile>;
    using D_gl       = gl<bf16,      1,  1, -1, -1, D_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;
    D_gl       D;

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
        int x = min((D.rows() / C::Mb) * (D.cols() / C::Nb), num_sms());
        if constexpr (C::CLUSTER_SIZE > 1) {
            int rem = x % C::CLUSTER_SIZE;
            if (rem != 0) x += C::CLUSTER_SIZE - rem;
        }
        return dim3(x);
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

template <typename C, bool B_PAYLOAD_TRANSPOSED = false, bool A_SCALE_SWAPPED = false>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::CLUSTER_SIZE == 2, "mxfp4_atb_gemm currently expects CTA-group 2");
    static_assert(C::Kb == 256, "mxfp4_atb_gemm currently expects Kb=256");

    if (threadIdx.x == 0) {
        g.A.template prefetch_tma<typename G::A_fp4x2_tile>();
        g.A_sc.template prefetch_tma<typename G::A_sc_tile>();
        g.B.template prefetch_tma<typename G::B_fp4x2_tile>();
        g.B_sc.template prefetch_tma<typename G::B_sc_tile>();
        g.D.template prefetch_tma<typename G::D_tile>();
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_row_blocks = g.D.rows() / C::Mb;
    const int num_col_blocks = g.D.cols() / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_iters_per_block = g.A.rows() / C::Kb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int *)&__shm[0]);
    typename G::input_tiles_t  (&input_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&input_scales)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t       &output_tiles                    = sm_allocator.allocate<typename G::outputs_t>();

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

    auto resolve_block = [&](int block_idx, int &row_block_idx, int &col_block_idx) {
        int supergroup_idx = block_idx / num_blocks_per_supergroup;
        int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
        int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
        int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
        row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
        col_block_idx = idx_within_supergroup / rows_in_supergroup;
    };

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS * C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            if constexpr (C::USE_PDL) pdl::wait();

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int row_block_idx, col_block_idx;
                resolve_block(block_idx, row_block_idx, col_block_idx);
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    // A and B are K x M/N, so both payloads are loaded in MN-major order:
                    // the row coordinate advances through reduction tiles, the column
                    // coordinate selects the output M/N tile.
                    tma::cluster::load_async(input_tiles[stage].A, g.A, {i * C::CLUSTER_SIZE + cta_id, row_block_idx}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    if constexpr (B_PAYLOAD_TRANSPOSED) {
                        tma::cluster::load_async(input_tiles[stage].B, g.B, {col_block_idx * C::CLUSTER_SIZE + cta_id, i}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    } else {
                        tma::cluster::load_async(input_tiles[stage].B, g.B, {i * C::CLUSTER_SIZE + cta_id, col_block_idx}, tiles_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    }
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            if constexpr (C::USE_PDL) pdl::wait();

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int row_block_idx, col_block_idx;
                resolve_block(block_idx, row_block_idx, col_block_idx);
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    #pragma unroll
                    for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                        if constexpr (A_SCALE_SWAPPED) {
                            tma::cluster::load_async(
                                input_scales[stage].A[k],
                                g.A_sc,
                                {i * C::MMA_PER_TILE + k, row_block_idx * C::CLUSTER_SIZE + cta_id, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(1 << cta_id),
                                0);
                        } else {
                            tma::cluster::load_async(
                                input_scales[stage].A[k],
                                g.A_sc,
                                {row_block_idx * C::CLUSTER_SIZE + cta_id, i * C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(1 << cta_id),
                                0);
                        }
                    }
                    if constexpr (C::B_SC_SIZE == 2) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                            tma::cluster::load_async(
                                input_scales[stage].B[cta_id * C::MMA_PER_TILE + k],
                                g.B_sc,
                                {col_block_idx * C::CLUSTER_SIZE + cta_id, i * C::MMA_PER_TILE + k, 0, 0},
                                scales_arrived[stage],
                                (uint16_t)(0b11),
                                0);
                        }
                    } else if (cta_id == 0) {
                        #pragma unroll
                        for (int k = 0; k < C::MMA_PER_TILE; ++k) {
                            tma::cluster::load_async(
                                input_scales[stage].B[k],
                                g.B_sc,
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
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                bool issued_mma = false;
                for (int i = 0; i < num_iters_per_block; ++i) {
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
                    auto A_sc_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16);
                    auto B_sc_subtile = B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32);
                    if (!issued_mma) {
                        if constexpr (B_PAYLOAD_TRANSPOSED) {
                            kittens::mma<transpose::T, transpose::T, decltype(out_tm), typename G::A_fp4x2_tile, typename G::B_fp4x2_tile, decltype(A_sc_subtile), decltype(B_sc_subtile), 0, C::CLUSTER_SIZE>(
                                out_tm,
                                input_tiles[stage].A,
                                input_tiles[stage].B,
                                A_sc_subtile,
                                B_sc_subtile,
                                inputs_finished[stage]);
                        } else {
                            kittens::mma<transpose::T, transpose::N, decltype(out_tm), typename G::A_fp4x2_tile, typename G::B_fp4x2_tile, decltype(A_sc_subtile), decltype(B_sc_subtile), 0, C::CLUSTER_SIZE>(
                                out_tm,
                                input_tiles[stage].A,
                                input_tiles[stage].B,
                                A_sc_subtile,
                                B_sc_subtile,
                                inputs_finished[stage]);
                        }
                        issued_mma = true;
                    } else {
                        if constexpr (B_PAYLOAD_TRANSPOSED) {
                            kittens::mma<transpose::T, transpose::T, decltype(out_tm), typename G::A_fp4x2_tile, typename G::B_fp4x2_tile, decltype(A_sc_subtile), decltype(B_sc_subtile), 1, C::CLUSTER_SIZE>(
                                out_tm,
                                input_tiles[stage].A,
                                input_tiles[stage].B,
                                A_sc_subtile,
                                B_sc_subtile,
                                inputs_finished[stage]);
                        } else {
                            kittens::mma<transpose::T, transpose::N, decltype(out_tm), typename G::A_fp4x2_tile, typename G::B_fp4x2_tile, decltype(A_sc_subtile), decltype(B_sc_subtile), 1, C::CLUSTER_SIZE>(
                                out_tm,
                                input_tiles[stage].A,
                                input_tiles[stage].B,
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
        if constexpr (C::USE_PDL) warpgroup::pdl::wait();
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int row_block_idx, col_block_idx;
            resolve_block(block_idx, row_block_idx, col_block_idx);
            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            constexpr float MXFP4_ALPHA = 1.0f / 36.0f;
            rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                warpgroup::load_async(D_reg_fl[i], out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                warp::mul(D_reg_fl[i], D_reg_fl[i], MXFP4_ALPHA);
            }
            rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                warp::copy(D_reg[i], D_reg_fl[i]);
            }
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i % C::NUM_D_TILES], D_reg[i]);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    g.D,
                    output_tiles.D[i % C::NUM_D_TILES],
                    {row_block_idx * C::CLUSTER_SIZE + cta_id, C::EPI_PIPE_DEPTH * col_block_idx + i});
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

} // namespace mxfp4_atb_gemm
