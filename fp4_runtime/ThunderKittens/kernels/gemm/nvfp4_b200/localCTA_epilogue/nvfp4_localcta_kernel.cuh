#pragma once
// ================================================================
// NVFP4 Local-CTA GEMM Kernel
// D = A x B^T with chunk-grid decode scales consumed in the K loop.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_localcta_gemm {

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH, int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI, int _Mb = 256, bool _USE_PDL = true, int _CLUSTER_SIZE = 2, int _Kb = 256>
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
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;
};

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<C::MMA_PER_TILE, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl    = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using B_fp4x2_gl = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl    = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using D_gl       = gl<bf16,       1,  1, -1, -1, D_tile>;

    A_fp4x2_gl A;
    A_sc_gl    A_sc;
    const float* A_sg_chunks;
    int          A_sg_stride;
    B_fp4x2_gl B;
    B_sc_gl    B_sc;
    const float* B_sg_chunks;
    int          B_sg_stride;
    D_gl        D;

    D_gl  D_K;
    D_gl  D_V;
    int   q_dim;
    int   k_dim;
    int   v_dim;
    bool  use_split_D;
    int   silu_dim;

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
        int d_cols = use_split_D ? (q_dim + k_dim + v_dim) : D.cols();
        int grid_size = min((D.rows()/(C::Mb/2))*(d_cols/C::Nb), num_sms());
        grid_size = (grid_size / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
        return dim3(max(grid_size, C::CLUSTER_SIZE));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int shm = sizeof(input_tiles_t)  * C::LOAD_PIPE_DEPTH + 1024 +
                            sizeof(input_scales_t) * C::LOAD_PIPE_DEPTH + 1024 +
                            sizeof(outputs_t);
        static_assert(shm <= MAX_SHARED_MEMORY - 1024);
        return shm;
    }
};

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

template <int ROWS, int COLS>
__device__ inline void scale_shared_fp8_tile_to_e8m0(st_fp8e4m3<ROWS, COLS, false> &tile, float scale) {
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
        const fp8e8m0_4 packed_out = base_types::convertor<fp8e8m0_4, float4>::convert(unpacked);
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
    const float *B_sg_chunks,
    int B_sg_stride,
    int a_chunk_row,
    int b_chunk_row_0,
    int chunk_base
) {
    #pragma unroll
    for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
        const int chunk_k = chunk_base + ii / 2;
        const float a_chunk_sg = A_sg_chunks[a_chunk_row * A_sg_stride + chunk_k];
        auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
            reinterpret_cast<uint64_t>(&A_sc_tile.data[0]) + 16 * 32 * ii);
        scale_shared_fp8_tile_to_e8m0(A_sc_sm_subtile, a_chunk_sg);

        const float b_chunk_sg_0 = B_sg_chunks[b_chunk_row_0 * B_sg_stride + chunk_k];
        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
            reinterpret_cast<uint64_t>(&B_sc_tiles[0].data[0]) + 16 * 32 * ii);
        scale_shared_fp8_tile_to_e8m0(B_sc_sm_subtile_0, b_chunk_sg_0);

        if constexpr (C::B_SC_SIZE == 2) {
            const float b_chunk_sg_1 = B_sg_chunks[(b_chunk_row_0 + 1) * B_sg_stride + chunk_k];
            auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&B_sc_tiles[1].data[0]) + 16 * 32 * ii);
            scale_shared_fp8_tile_to_e8m0(B_sc_sm_subtile_1, b_chunk_sg_1);
        }
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

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
    __shared__ semaphore scale_tiles_ready[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_prepared[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_provisioned, 0, 1);
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
    }
    everyone::tma::cluster::arrive_aligned();

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        const int lane = threadIdx.x % WARP_THREADS;
        if (warp_id == 3 && warp::elect_leader()) {
                if constexpr (C::USE_PDL) pdl::wait();
                everyone::tma::cluster::wait();
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
                if constexpr (C::USE_PDL) pdl::wait();
                everyone::tma::cluster::wait();
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
        } else if (warp_id == 1) {
                everyone::tma::cluster::wait();
                uint32_t ready_phasebits = 0;
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                    int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                    int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                    for (int i = 0; i < num_red_blocks; ++i) {
                        const int chunk_base = i * (C::Kb / 128);
                        const int a_chunk_row = row_block_idx * 2 + cta_id;
                        const int b_chunk_row_0 = col_block_idx * C::B_SC_SIZE;

                        if (cta_id == 0) {
                            if (lane == 0) {
                                tma::expect_bytes(scales_arrived[stage], 2 * sizeof(G::input_scales_t));
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

                        apply_chunk_scales_to_stage<C>(
                            input_scales[stage].A, input_scales[stage].B,
                            g.A_sg_chunks, g.A_sg_stride,
                            g.B_sg_chunks, g.B_sg_stride,
                            a_chunk_row, b_chunk_row_0, chunk_base);

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
        } else if (cta_id == 0 && warp_id == 0 && warp::elect_leader()) {
                everyone::tma::cluster::wait();
                wait(tmem_provisioned, 0);
                tm_allocator.set_addr(tmem_addr);
                auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
                auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
                auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e8m0<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256 + 4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
                uint32_t tensor_phasebits = 0xFFFF0000;
                uint32_t prepared_phasebits = 0;
                for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                    int supergroup_idx = block_idx / num_blocks_per_supergroup;
                    int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                    int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                    int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                    int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                    int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                    wait(outputs_finished, get_phasebit<1>(tensor_phasebits, 0));
                    tensor_after_thread_sync();

                    for (int i = 0; i < num_red_blocks; ++i) {
                        wait(scales_prepared[stage], get_phasebit<0>(prepared_phasebits, stage));

                        #pragma unroll
                        for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                            auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e8m0<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);

                            auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16);
                            auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e8m0<32, 16, false> *>(
                                reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);

                            if constexpr (C::B_SC_SIZE == 2) {
                                auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e8m0<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16 + 16);
                                auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e8m0<32, 16, false> *>(
                                    reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                            }
                        }
                        update_phasebit<0>(prepared_phasebits, stage);

                        tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(tensor_phasebits, stage));
                        if (i == 0) {
                            mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                    A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                    B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                    inputs_finished[stage]);
                        } else {
                            mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                     A_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                     B_sc_tm.template subtile<full_tt_fp8e8m0<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                     inputs_finished[stage]);
                        }
                        update_phasebit<0>(tensor_phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                    tensor_commit<2>(outputs_arrived);
                    update_phasebit<1>(tensor_phasebits, 0);
                }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
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
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

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
                    if (g.silu_dim > 0) {
                        int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                        if (col_offset_elems < g.silu_dim) {
                            apply_silu_inplace(D_reg);
                        }
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
                    if (g.silu_dim > 0) {
                        int col_offset_elems = (C::EPI_PIPE_DEPTH*col_block_idx + i) * C::Nb/C::EPI_PIPE_DEPTH;
                        if (col_offset_elems < g.silu_dim) {
                            apply_silu_inplace(D_reg_fl);
                        }
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
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }
}

} // namespace nvfp4_localcta_gemm
