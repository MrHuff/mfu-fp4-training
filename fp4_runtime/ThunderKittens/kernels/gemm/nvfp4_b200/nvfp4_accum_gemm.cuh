#pragma once
// ================================================================
// NVFP4 Batched GEMM with Accumulating Consumer Epilogue
//
// All batches accumulate into a single D_out buffer, eliminating
// the separate sum3 kernel. Batch 0 stores via TMA as usual;
// batches 1+ wait for predecessor, TMA-load the old value from
// D_out into SMEM, add in registers, and TMA-store the sum.
//
// Per-tile completion flags (`tile_done`) enforce ordering:
//   tile_done[tile_idx] starts at 0
//   batch b waits for tile_done[tile_idx] >= b before reading
//   batch b increments tile_done[tile_idx] after writing
//
// Note: this reuses the existing output_tiles.D SMEM buffer.
// For batches 1+, we serialize: load old → add → store new.
// The EPI_PIPE_DEPTH sub-tiles are processed sequentially.
// ================================================================

#include "nvfp4_gemm.cuh"
#include "nvfp4_batched_gemm.cuh"

using namespace kittens;

namespace nvfp4_accum_gemm {

using nvfp4_batched_gemm::tma_dev_proxy;

static constexpr int MAX_BATCHES = nvfp4_batched_gemm::MAX_BATCHES;

template <typename C>
struct globals {
    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;

    using A_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,       1, -1, -1, 256, A_sc_tile>;
    using B_fp4x2_gl     = gl<fp4e2m1_2,  1,  1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl        = gl<half,       1, -1, -1, 256, B_sc_tile>;
    using D_gl           = gl<bf16,       1,  1, -1, -1, D_tile>;

    CUtensorMap A_tma[MAX_BATCHES];
    CUtensorMap A_sc_tma[MAX_BATCHES];
    CUtensorMap B_tma[MAX_BATCHES];
    CUtensorMap B_sc_tma[MAX_BATCHES];

    // Single shared D output (all batches write here)
    CUtensorMap D_tma;

    const float* A_sg[MAX_BATCHES];
    const float* B_sg[MAX_BATCHES];
    int          A_sg_stride[MAX_BATCHES];
    int          B_sg_stride[MAX_BATCHES];

    int       num_red_blocks[MAX_BATCHES];
    int       num_batches;
    int       num_row_blocks;
    int       num_col_blocks;

    // Per-tile completion counter
    // tile_done[row_block*2 + cta_id][col_block] counts completed batches
    int*      tile_done;

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
        return dim3(min(spatial_tiles, num_sms()), 1, num_batches);
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
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;

    const int batch = blockIdx.z;
    const int num_blocks = g.num_row_blocks * g.num_col_blocks;

    // Prefetch this batch's TMA descriptors
    if (threadIdx.x == 0) {
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_tma[batch])) : "memory");
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.A_sc_tma[batch])) : "memory");
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_tma[batch])) : "memory");
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.B_sc_tma[batch])) : "memory");
        asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(&g.D_tma)) : "memory");
    }

    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_red_blocks = g.num_red_blocks[batch];
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
    // Extra semaphore for TMA load of old D_out (used by batch > 0)
    __shared__ semaphore accum_load_arrived;
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
        init_semaphore(accum_load_arrived, 0, 1);
    }
    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait_aligned();

    // Create proxies for this batch's TMA descriptors
    tma_dev_proxy<typename G::A_fp4x2_gl> proxy_A(&g.A_tma[batch]);
    tma_dev_proxy<typename G::B_fp4x2_gl> proxy_B(&g.B_tma[batch]);
    tma_dev_proxy<typename G::A_sc_gl>    proxy_A_sc(&g.A_sc_tma[batch]);
    tma_dev_proxy<typename G::B_sc_gl>    proxy_B_sc(&g.B_sc_tma[batch]);
    // All batches share the same D output proxy
    tma_dev_proxy<typename G::D_gl>       proxy_D(&g.D_tma);

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            // ── Producer: load input tiles ── (identical to batched GEMM)
            if constexpr (C::USE_PDL) pdl::wait();

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_tiles[stage].A, proxy_A, {row_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(input_tiles[stage].B, proxy_B, {col_block_idx*2 + cta_id, i}, tiles_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            // ── Producer: load input scales ── (identical to batched GEMM)
            if constexpr (C::USE_PDL) pdl::wait();

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(input_scales[stage].A, proxy_A_sc, {row_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(1<<cta_id), 0);
                    if constexpr (C::B_SC_SIZE == 2) tma::cluster::load_async(input_scales[stage].B[cta_id], proxy_B_sc, {col_block_idx*2 + cta_id, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    else if (cta_id == 0)            tma::cluster::load_async(input_scales[stage].B[0], proxy_B_sc, {col_block_idx, i, 0}, scales_arrived[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            // ── MMA warp ── (identical to batched GEMM)
            wait(tmem_provisioned, 0);
            tm_allocator.set_addr(tmem_addr);
            auto out_tm  = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_allocator.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
                wait(outputs_finished, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_red_blocks; i++) {
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
                        if constexpr (C::B_SC_SIZE == 2) {
                            auto B_sc_tm_subtile_1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16+16);
                            auto &B_sc_sm_subtile_1 = *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(reinterpret_cast<uint64_t>(&input_scales[stage].B[1].data[0])+16*32*ii);
                            load_mxnv_scale_async2(B_sc_tm_subtile_1, B_sc_sm_subtile_1);
                        }
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
        // ── Consumer: scale and ACCUMULATE into shared D output ──
        if (warpgroup::warpid() == 0) {
            tm_allocator.provision(tmem_addr);
            warp::arrive(tmem_provisioned);
        }
        wait(tmem_provisioned, 0);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        int accum_mbar_phase = 0;

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(C::SUPERGROUP_SIZE, g.num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;
            const float global_scale =
                g.A_sg[batch][row_block_idx * g.A_sg_stride[batch]] *
                g.B_sg[batch][col_block_idx * g.B_sg_stride[batch]];

            wait(outputs_arrived, get_phasebit<0>(phasebits, 0));

            // Load MMA results from TMEM, scale, convert to bf16
            rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg[C::EPI_PIPE_DEPTH];
            #pragma unroll
            for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                rt_fl<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_reg_fl;
                warpgroup::load_async(D_reg_fl, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                warp::mul(D_reg_fl, D_reg_fl, global_scale);
                warp::copy(D_reg[i], D_reg_fl);
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(1);
            warpgroup::tma::cluster::arrive(outputs_finished, 0, 1);

            // Compute global tile index for semaphore
            int tile_idx = (row_block_idx * 2 + cta_id) * g.num_col_blocks + col_block_idx;

            if (batch > 0) {
                // ─── Accumulate: wait for predecessor, load old, add, store ───
                // Spin-wait for previous batch to finish this tile.
                // Only leader polls; warpgroup::sync broadcasts to consumer threads.
                // NOTE: cannot use __syncthreads() here — producer warpgroup is in
                // elect_leader() path and would deadlock.
                if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                    while (atomicAdd(&g.tile_done[tile_idx], 0) < batch) {
                        __nanosleep(32);
                    }
                }
                warpgroup::sync(1);

                // Process each epilogue sub-tile: load old → add → store
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    int smem_slot = i % C::NUM_D_TILES;

                    // TMA load old D_out tile into SMEM
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);

                    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                        tma::expect_bytes(accum_load_arrived, sizeof(typename G::D_tile));
                        tma::load_async(output_tiles.D[smem_slot], proxy_D,
                                       {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i},
                                       accum_load_arrived);
                    }
                    wait(accum_load_arrived, accum_mbar_phase);
                    accum_mbar_phase ^= 1;
                    warpgroup::sync(1);

                    // Load old bf16 values from SMEM → registers and add
                    rt_bf<C::Mb / 8, C::Nb/C::EPI_PIPE_DEPTH> D_old;
                    warpgroup::load(D_old, output_tiles.D[smem_slot]);
                    warpgroup::sync(1);

                    // Add: D_reg[i] += D_old (element-wise bf16 add)
                    warp::add(D_reg[i], D_reg[i], D_old);

                    // Store sum to SMEM and TMA store to GMEM
                    warpgroup::store(output_tiles.D[smem_slot], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(proxy_D, output_tiles.D[smem_slot], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
                }
            } else {
                // ─── Batch 0: normal store (no load needed) ───
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>();
                    warpgroup::sync(1);
                    warpgroup::store(output_tiles.D[i%C::NUM_D_TILES], D_reg[i]);
                    warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(proxy_D, output_tiles.D[i%C::NUM_D_TILES], {row_block_idx*2 + cta_id, C::EPI_PIPE_DEPTH*col_block_idx + i});
                }
            }

            // All pending TMA stores must complete before signaling
            warpgroup::sync(1);
            warpgroup::tma::store_async_read_wait<0>();
            // Fence to ensure TMA stores are visible to other CTAs (batches)
            asm volatile("fence.proxy.async.global;\n" :::"memory");
            __threadfence();  // ensure stores are globally visible
            if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
                atomicAdd(&g.tile_done[tile_idx], 1);
            }

            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1);
        warpgroup::tma::store_async_read_wait<0>();
        if constexpr (C::USE_PDL) warpgroup::pdl::arrive();
        if (warpgroup::warpid() == 0) tm_allocator.deprovision();
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n");
    asm volatile("barrier.cluster.wait.aligned;\n");
}

} // namespace nvfp4_accum_gemm
