#pragma once
// ================================================================
// NVFP4 Split2 One-Pass Accum GEMM
// Specialized for 2-way concatenated dgrad:
//   D = A0 x B0^T + A1 x B1^T
// where A_i are strided slices of one full row-prepared A buffer.
// ================================================================

#include "nvfp4_gemm.cuh"

namespace nvfp4_split2_accum_gemm {

static constexpr int NUM_SPLITS = 2;

template <typename _GL>
struct tma_dev_proxy {
    using identifier = ducks::gl::identifier;
    using T     = typename _GL::T;
    using T2    = typename _GL::T2;
    using dtype = typename _GL::dtype;
    static constexpr int __b__ = _GL::__b__, __d__ = _GL::__d__, __r__ = _GL::__r__, __c__ = _GL::__c__;

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

    CUtensorMap A_tma[NUM_SPLITS];
    CUtensorMap A_sc_tma[NUM_SPLITS];
    CUtensorMap B_tma[NUM_SPLITS];
    CUtensorMap B_sc_tma[NUM_SPLITS];
    const float* A_sg[NUM_SPLITS];
    const float* B_sg[NUM_SPLITS];
    int A_sg_stride[NUM_SPLITS];
    int B_sg_stride[NUM_SPLITS];
    CUtensorMap D_tma;

    int num_red_blocks[NUM_SPLITS];
    int num_row_blocks;
    int num_col_blocks;

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
        int spatial_tiles = num_row_blocks * num_col_blocks;
        int grid_size = min(spatial_tiles, num_sms());
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
        scale_shared_fp8_tile(A_sc_sm_subtile, a_chunk_sg);

        const float b_chunk_sg_0 = B_sg_chunks[b_chunk_row_0 * B_sg_stride + chunk_k];
        auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
            reinterpret_cast<uint64_t>(&B_sc_tiles[0].data[0]) + 16 * 32 * ii);
        scale_shared_fp8_tile(B_sc_sm_subtile_0, b_chunk_sg_0);

        if constexpr (C::B_SC_SIZE == 2) {
            const float b_chunk_sg_1 = B_sg_chunks[(b_chunk_row_0 + 1) * B_sg_stride + chunk_k];
            auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                reinterpret_cast<uint64_t>(&B_sc_tiles[1].data[0]) + 16 * 32 * ii);
            scale_shared_fp8_tile(B_sc_sm_subtile_1, b_chunk_sg_1);
        }
    }
}

template <typename C, bool USE_CHUNK_GRID_SCALE>
__device__ inline void kernel_impl(const globals<C> &g) {
    using G = globals<C>;

    const int num_blocks = g.num_row_blocks * g.num_col_blocks;
    const bool bypass_outer_scale =
        g.A_sg_stride[0] == 0 && g.B_sg_stride[0] == 0 &&
        g.A_sg_stride[1] == 0 && g.B_sg_stride[1] == 0 &&
        g.A_sg[0] == g.B_sg[0] && g.A_sg[1] == g.B_sg[1];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int batch = 0; batch < NUM_SPLITS; ++batch) {
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[batch])) : "memory");
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
                for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                    tma_dev_proxy<typename G::A_fp4x2_gl> proxy_A(&g.A_tma[batch]);
                    tma_dev_proxy<typename G::B_fp4x2_gl> proxy_B(&g.B_tma[batch]);
                    for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
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
                for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                    tma_dev_proxy<typename G::A_sc_gl> proxy_A_sc(&g.A_sc_tma[batch]);
                    tma_dev_proxy<typename G::B_sc_gl> proxy_B_sc(&g.B_sc_tma[batch]);
                    constexpr uint16_t b_mask = C::CLUSTER_SIZE == 2 ? uint16_t(0b11) : uint16_t(0b1);
                    for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        tma::cluster::load_async(input_scales[stage].A, proxy_A_sc, {row_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        if constexpr (C::B_SC_SIZE == 2) {
                            tma::cluster::load_async(input_scales[stage].B[cta_id], proxy_B_sc, {col_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], b_mask, 0);
                        } else if (cta_id == 0) {
                            tma::cluster::load_async(input_scales[stage].B[0], proxy_B_sc, {col_block_idx, i, 0}, scales_arrived[stage], b_mask, 0);
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                if (bypass_outer_scale || USE_CHUNK_GRID_SCALE) {
                    wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                    tensor_after_thread_sync();
                    bool first_mma = true;
                    #pragma unroll
                    for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                        for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
                            tma::expect_bytes(scales_arrived[stage], 2 * sizeof(typename G::input_scales_t));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                            if constexpr (USE_CHUNK_GRID_SCALE) {
                                const int chunk_base = i * (C::Kb / 128);
                                const int a_chunk_row = row_block_idx * 2 + cta_id;
                                const int b_chunk_row_0 = col_block_idx * C::B_SC_SIZE;
                                const int A_sg_stride = g.A_sg_stride[batch] < 0 ? -g.A_sg_stride[batch] : g.A_sg_stride[batch];
                                const int B_sg_stride = g.B_sg_stride[batch] < 0 ? -g.B_sg_stride[batch] : g.B_sg_stride[batch];
                                apply_chunk_scales_to_stage<C>(
                                    input_scales[stage].A, input_scales[stage].B,
                                    g.A_sg[batch], A_sg_stride,
                                    g.B_sg[batch], B_sg_stride,
                                    a_chunk_row, b_chunk_row_0, chunk_base);
                                __syncwarp();
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            }
                            #pragma unroll
                            for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                                auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                                auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                                auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16);
                                auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                                if constexpr (C::B_SC_SIZE == 2) {
                                    auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16 + 16);
                                    auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                                    load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                                }
                            }
                            tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(typename G::input_tiles_t));
                            wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                            if (first_mma) {
                                mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                        inputs_finished[stage]);
                                first_mma = false;
                            } else {
                                mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                         A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                         B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                         inputs_finished[stage]);
                            }
                            update_phasebit<0>(phasebits, stage);
                            stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        }
                    }
                    tensor_commit<2>(outputs_arrived);
                    update_phasebit<1>(phasebits, 0);
                } else {
                    #pragma unroll
                    for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                        wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                        tensor_after_thread_sync();
                        bool first_mma = true;
                        for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
                            tma::expect_bytes(scales_arrived[stage], 2 * sizeof(typename G::input_scales_t));
                            wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                            #pragma unroll
                            for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                                auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                                auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                                auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16);
                                auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                                if constexpr (C::B_SC_SIZE == 2) {
                                    auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16 + 16);
                                    auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                                    load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                                }
                            }
                            tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(typename G::input_tiles_t));
                            wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                            if (first_mma) {
                                mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                        A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                        B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                        inputs_finished[stage]);
                                first_mma = false;
                            } else {
                                mma2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                         A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                         B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
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
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
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

            if (bypass_outer_scale) {
                wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                    rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                    warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(0, C::Nb / C::EPI_PIPE_DEPTH * i));
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warp::copy(D_reg[i], D_reg_fl);
                }
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i % C::NUM_D_TILES], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(proxy_D, output_tiles.D[i % C::NUM_D_TILES], {row_block_idx * 2 + cta_id, C::EPI_PIPE_DEPTH * col_block_idx + i});
                }
                update_phasebit<0>(phasebits, 0);
            } else {
                rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_acc[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                    wait(outputs_arrived, get_phasebit<0>(phasebits, 0));
                    constexpr int ROW_SG_TILE = 256;
                    constexpr int COL_SG_TILE = 256;
                    const int row_sg_idx = row_block_idx / (ROW_SG_TILE / C::Mb);
                    const int col_sg_idx = col_block_idx / (COL_SG_TILE / C::Nb);
                    const float global_scale =
                        g.A_sg[batch][row_sg_idx * g.A_sg_stride[batch]] *
                        g.B_sg[batch][col_sg_idx * g.B_sg_stride[batch]];
                    #pragma unroll
                    for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                        rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg_fl;
                        warpgroup::load_async(
                            D_reg_fl,
                            out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                0, C::Nb / C::EPI_PIPE_DEPTH * i));
                        tensor_load_wait();
                        tensor_before_thread_sync();
                        warpgroup::sync(1);
                        warp::mul(D_reg_fl, D_reg_fl, global_scale);
                        if (batch == 0) warp::copy(D_acc[i], D_reg_fl);
                        else            warp::add(D_acc[i], D_acc[i], D_reg_fl);
                        warpgroup::sync(1);
                        tensor_after_thread_sync();
                    }
                    warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);
                    update_phasebit<0>(phasebits, 0);
                }
                rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                    warp::copy(D_reg[i], D_acc[i]);
                }
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i % C::NUM_D_TILES], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(proxy_D, output_tiles.D[i % C::NUM_D_TILES], {row_block_idx * 2 + cta_id, C::EPI_PIPE_DEPTH * col_block_idx + i});
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

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    kernel_impl<C, false>(g);
}

template <typename C>
__device__ inline void kernel_chunk_grid(const globals<C> &g) {
    using G = globals<C>;
    static_assert(C::Kb == 128, "chunk-grid v4 split2 consumer applies one fp32 SG per 128-wide reduction chunk");

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int batch = 0; batch < NUM_SPLITS; ++batch) {
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[batch])) : "memory");
            asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[batch])) : "memory");
        }
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.D_tma)) : "memory");
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_blocks = g.num_row_blocks * g.num_col_blocks;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * g.num_col_blocks;
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
                for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                    tma_dev_proxy<typename G::A_fp4x2_gl> proxy_A(&g.A_tma[batch]);
                    tma_dev_proxy<typename G::B_fp4x2_gl> proxy_B(&g.B_tma[batch]);
                    for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
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
                for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                    tma_dev_proxy<typename G::A_sc_gl> proxy_A_sc(&g.A_sc_tma[batch]);
                    tma_dev_proxy<typename G::B_sc_gl> proxy_B_sc(&g.B_sc_tma[batch]);
                    constexpr uint16_t b_mask = C::CLUSTER_SIZE == 2 ? uint16_t(0b11) : uint16_t(0b1);
                    for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        tma::cluster::load_async(input_scales[stage].A, proxy_A_sc, {row_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1 << cta_id), 0);
                        if constexpr (C::B_SC_SIZE == 2) {
                            tma::cluster::load_async(input_scales[stage].B[cta_id], proxy_B_sc, {col_block_idx * 2 + cta_id, i, 0}, scales_arrived[stage], b_mask, 0);
                        } else if (cta_id == 0) {
                            tma::cluster::load_async(input_scales[stage].B[0], proxy_B_sc, {col_block_idx, i, 0}, scales_arrived[stage], b_mask, 0);
                        }
                        update_phasebit<1>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                    }
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto out_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(128);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH>>(256 + 4 * C::MMA_PER_TILE * C::LOAD_PIPE_DEPTH);
            uint32_t output_phasebits = 0xFFFF0000;

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int output_idx = 0;
                #pragma unroll
                for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                    for (int i = 0; i < g.num_red_blocks[batch]; ++i) {
                        const int output_slot = OUTPUT_PIPE_DEPTH == 2 ? (output_idx & 1) : 0;
                        auto &out_tm = output_slot == 0 ? out_tm_0 : out_tm_1;
                        wait(outputs_finished[output_slot], get_phasebit<1>(output_phasebits, output_slot));
                        tensor_after_thread_sync();

                        tma::expect_bytes(scales_arrived[stage], 2 * sizeof(typename G::input_scales_t));
                        wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                        #pragma unroll
                        for (int ii = 0; ii < C::MMA_PER_TILE; ++ii) {
                            auto A_sc_tm_subtile = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 16 + ii * 16);
                            auto &A_sc_sm_subtile = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].A.data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(A_sc_tm_subtile, A_sc_sm_subtile);
                            auto B_sc_tm_subtile_0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16);
                            auto &B_sc_sm_subtile_0 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[0].data[0]) + 16 * 32 * ii);
                            load_mxnv_scale_async2(B_sc_tm_subtile_0, B_sc_sm_subtile_0);
                            if constexpr (C::B_SC_SIZE == 2) {
                                auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage * C::MMA_PER_TILE * 32 + ii * C::B_SC_SIZE * 16 + 16);
                                auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0]) + 16 * 32 * ii);
                                load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                            }
                        }
                        tma::expect_bytes(tiles_arrived[stage], 2 * sizeof(typename G::input_tiles_t));
                        wait(tiles_arrived[stage], get_phasebit<0>(phasebits, stage));
                        mm2_ABt(out_tm, input_tiles[stage].A, input_tiles[stage].B,
                                A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 16>>(stage * C::MMA_PER_TILE * 16),
                                B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE * 32>>(stage * C::MMA_PER_TILE * 32),
                                inputs_finished[stage]);
                        tensor_commit<2>(outputs_arrived[output_slot]);
                        update_phasebit<1>(output_phasebits, output_slot);
                        update_phasebit<0>(phasebits, stage);
                        stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                        ++output_idx;
                    }
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

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_acc[C::EPI_PIPE_DEPTH];
            int output_idx = 0;
            {
                constexpr int batch = 0;
                constexpr int i = 0;
                const int output_slot = 0;
                auto &out_tm = out_tm_0;
                wait(outputs_arrived[output_slot], get_phasebit<0>(output_phasebits, output_slot));

                const int A_sg_stride = g.A_sg_stride[batch] < 0 ? -g.A_sg_stride[batch] : g.A_sg_stride[batch];
                const int B_sg_stride = g.B_sg_stride[batch] < 0 ? -g.B_sg_stride[batch] : g.B_sg_stride[batch];
                const int a_chunk_row = row_block_idx * C::CLUSTER_SIZE + cta_id;
                const float a_sg = g.A_sg[batch][a_chunk_row * A_sg_stride + i];
                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    warpgroup::load_async(
                        D_acc[epi],
                        out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                            0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                }
                tensor_load_wait();
                tensor_before_thread_sync();
                warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(outputs_finished[output_slot], 0, 1);

                #pragma unroll
                for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                    constexpr int cols_per_epi = C::Nb / C::EPI_PIPE_DEPTH;
                    const int b_chunk_col = col_block_idx * (C::Nb / 128) + (epi * cols_per_epi) / 128;
                    const float b_sg = g.B_sg[batch][b_chunk_col * B_sg_stride + i];
                    const float gs = a_sg * b_sg;
                    warp::mul(D_acc[epi], D_acc[epi], gs);
                }
                warpgroup::sync(1);
                tensor_after_thread_sync();
                update_phasebit<0>(output_phasebits, output_slot);
                ++output_idx;
            }

            #pragma unroll
            for (int batch = 0; batch < NUM_SPLITS; ++batch) {
                const int A_sg_stride = g.A_sg_stride[batch] < 0 ? -g.A_sg_stride[batch] : g.A_sg_stride[batch];
                const int B_sg_stride = g.B_sg_stride[batch] < 0 ? -g.B_sg_stride[batch] : g.B_sg_stride[batch];
                const int first_i = batch == 0 ? 1 : 0;
                for (int i = first_i; i < g.num_red_blocks[batch]; ++i) {
                    const int output_slot = OUTPUT_PIPE_DEPTH == 2 ? (output_idx & 1) : 0;
                    auto &out_tm = output_slot == 0 ? out_tm_0 : out_tm_1;
                    wait(outputs_arrived[output_slot], get_phasebit<0>(output_phasebits, output_slot));

                    const int a_chunk_row = row_block_idx * C::CLUSTER_SIZE + cta_id;
                    const float a_sg = g.A_sg[batch][a_chunk_row * A_sg_stride + i];
                    rt_fl<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_pipe[C::EPI_PIPE_DEPTH];
                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                        warpgroup::load_async(
                            D_pipe[epi],
                            out_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                                0, C::Nb / C::EPI_PIPE_DEPTH * epi));
                    }
                    tensor_load_wait();
                    tensor_before_thread_sync();
                    warpgroup::sync(1);
                    warpgroup::tma::cluster::arrive(outputs_finished[output_slot], 0, 1);

                    #pragma unroll
                    for (int epi = 0; epi < C::EPI_PIPE_DEPTH; ++epi) {
                        constexpr int cols_per_epi = C::Nb / C::EPI_PIPE_DEPTH;
                        const int b_chunk_col = col_block_idx * (C::Nb / 128) + (epi * cols_per_epi) / 128;
                        const float b_sg = g.B_sg[batch][b_chunk_col * B_sg_stride + i];
                        const float gs = a_sg * b_sg;
                        nvfp4_gemm::add_scaled_inplace(D_acc[epi], D_pipe[epi], gs);
                    }
                    warpgroup::sync(1);
                    tensor_after_thread_sync();
                    update_phasebit<0>(output_phasebits, output_slot);
                    ++output_idx;
                }
            }

            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; ++i) {
                rt_bf<C::Mb / 8, C::Nb / C::EPI_PIPE_DEPTH> D_reg;
                warp::copy(D_reg, D_acc[i]);
                warpgroup::tma::store_async_read_wait<C::NUM_D_TILES - 1>();
                warpgroup::sync(1);
                warpgroup::store(output_tiles.D[i % C::NUM_D_TILES], D_reg);
                warpgroup::sync(1);
                warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(
                    proxy_D, output_tiles.D[i % C::NUM_D_TILES],
                    {row_block_idx * 2 + cta_id, C::EPI_PIPE_DEPTH * col_block_idx + i});
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

} // namespace nvfp4_split2_accum_gemm
