#pragma once
// ================================================================
// Persistent Quantize→GEMM: single kernel launch.
// Phase 1: quantize bf16 A,B → FP4+scales in HBM
//   - CTA-local amax: per-16-element-group scaling (no global amax)
//   - Full 256-thread occupancy: both warpgroups quantize in parallel
// Phase 2: standard FP4 GEMM
// Grid barrier between phases.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_persistent_gemm {

template <int _Nb, int _LOAD_PIPE_DEPTH, int _EPI_PIPE_DEPTH,
          int _SUPERGROUP_SIZE, int _NUM_D_TILES, bool _OVERLAP_EPI,
          int _Mb = 256, bool _USE_PDL = false, int _CLUSTER_SIZE = 2>
struct config {
    static_assert(_Nb == 128 || _Nb == 256);
    static_assert(_Mb == 256 || _Mb == 512);
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
    static constexpr int Kb = 256;
    static constexpr int B_SC_SIZE = Nb / 128;
    static constexpr int MMA_PER_TILE = Kb / 64;
    static constexpr int NUM_D_TILES = _NUM_D_TILES;
    static constexpr auto D_CACHE_POLICY = cache_policy::EVICT_FIRST;
    static constexpr int QM = 128;
    static constexpr int QN = 128;
    static constexpr int QK = 16;
};

template <typename C>
struct globals {
    using Q_bf16_tile  = st_bf<C::QM, C::QN, false>;
    using Q_fp4_tile   = st_fp4e2m1_2<C::QM, C::QN/2, false>;
    using Q_sc_vec     = sv_hf<256>;
    using Q_bf16_gl    = gl<bf16, 1, 1, -1, -1, Q_bf16_tile>;
    using Q_fp4_gl     = gl<fp4e2m1_2, 1, 1, -1, -1, Q_fp4_tile>;
    using Q_sc_gl      = gl<half, 1, -1, -1, 256, Q_sc_vec>;

    using A_fp4x2_tile = st_fp4e2m1_2<C::Mb/2, C::Kb/2>;
    using A_sc_tile    = st_hf<4, 256, false>;
    using B_fp4x2_tile = st_fp4e2m1_2<C::Nb/2, C::Kb/2>;
    using B_sc_tile    = st_hf<4, 256, false>;
    using D_tile       = st_bf<C::Mb/2, C::Nb/C::EPI_PIPE_DEPTH>;
    using A_fp4x2_gl   = gl<fp4e2m1_2, 1, 1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl      = gl<half, 1, -1, -1, 256, A_sc_tile>;
    using A_sc_global_gl = gl<float, 1, 1, 1, 1>;
    using B_fp4x2_gl   = gl<fp4e2m1_2, 1, 1, -1, -1, B_fp4x2_tile>;
    using B_sc_gl      = gl<half, 1, -1, -1, 256, B_sc_tile>;
    using B_sc_global_gl = gl<float, 1, 1, 1, 1>;
    using D_gl         = gl<bf16, 1, 1, -1, -1, D_tile>;

    Q_bf16_gl A_bf16;
    Q_fp4_gl  A_q_fp4;
    Q_sc_gl   A_q_sc;
    Q_bf16_gl B_bf16;
    Q_fp4_gl  B_q_fp4;
    Q_sc_gl   B_q_sc;
    bool      quantize_b;

    A_fp4x2_gl     A;
    A_sc_gl        A_sc;
    A_sc_global_gl A_sc_global;
    B_fp4x2_gl     B;
    B_sc_gl        B_sc;
    B_sc_global_gl B_sc_global;
    D_gl           D;

    int* barrier;

    struct input_tiles_t  { A_fp4x2_tile A; B_fp4x2_tile B; };
    struct input_scales_t { A_sc_tile A; B_sc_tile B[C::B_SC_SIZE]; };
    struct outputs_t      { D_tile D[C::NUM_D_TILES]; };

    __host__ inline dim3 grid() const {
        return dim3(min((D.rows()/(C::Mb/2))*(D.cols()/C::Nb), num_sms()));
    }
    __host__ inline dim3 block() const { return dim3(C::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() const {
        constexpr int gemm_smem = sizeof(input_tiles_t)*C::LOAD_PIPE_DEPTH + 1024 +
                                  sizeof(input_scales_t)*C::LOAD_PIPE_DEPTH + 1024 +
                                  sizeof(outputs_t);
        constexpr int quant_smem = 2 * C::QM * C::QN * sizeof(bf16) + 2048;
        constexpr int smem = (gemm_smem > quant_smem) ? gemm_smem : quant_smem;
        static_assert(smem <= MAX_SHARED_MEMORY - 1024);
        return smem;
    }
};

// Grid barrier via volatile spin. Safe for persistent grids only.
__device__ inline void grid_barrier(int* counter, int num_ctas) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        unsigned int old = atomicAdd((unsigned int*)counter, 1u);
        volatile unsigned int* vc = (volatile unsigned int*)counter;
        while (*vc < (unsigned int)num_ctas) {}
    }
    __syncthreads();
}

// ================================================================
// Phase 1: Quantize one operand with per-block amax.
// Pipelined: 2 SMEM stages, overlap TMA load of next tile with compute.
// 128 threads active for compute, all 256 participate in syncs.
// ================================================================
template <typename C, typename G>
__device__ inline void quantize_operand(
    const typename G::Q_bf16_gl &bf16_in,
    const typename G::Q_fp4_gl &fp4_out,
    const typename G::Q_sc_gl  &sc_out
) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_alloc((int*)&__shm[0]);
    // 2 pipeline stages for overlapping load+compute
    auto &bf16_smem0 = sm_alloc.allocate<typename G::Q_bf16_tile>();
    auto &bf16_smem1 = sm_alloc.allocate<typename G::Q_bf16_tile>();
    typename G::Q_bf16_tile* bf16_stages[2] = {&bf16_smem0, &bf16_smem1};

    const int tid = threadIdx.x;
    const bool active = (tid < 128);
    const int tile_row = tid;  // Only valid for tid < 128

    constexpr float S_DEC = 65504.0f / (6.0f * 448.0f);
    constexpr float S_ENC = 1.0f / S_DEC;
    constexpr int NKH = C::QN / C::QK / 2;
    constexpr int NPK = C::QK / 2;

    const int num_row_tiles = bf16_in.rows() / C::QM;
    const int num_col_tiles = bf16_in.cols() / C::QN;
    const int num_tiles = num_row_tiles * num_col_tiles;

    __shared__ semaphore q_arrived[2];
    if (tid == 0) {
        init_semaphore(q_arrived[0], 0, 1);
        init_semaphore(q_arrived[1], 0, 1);
    }
    __syncthreads();

    // Build list of tiles this CTA handles
    int first_t = blockIdx.x;
    if (first_t >= num_tiles) return;

    // Kick off first TMA load
    if (tid == 0) {
        tma::expect(q_arrived[0], *bf16_stages[0]);
        int r = first_t / num_col_tiles, c = first_t % num_col_tiles;
        tma::load_async(*bf16_stages[0], bf16_in, {r, c}, q_arrived[0]);
    }

    int phase0 = 0, phase1 = 0;
    int stage = 0;

    for (int t = first_t; t < num_tiles; t += gridDim.x) {
        const int row = t / num_col_tiles;
        const int col = t % num_col_tiles;
        const int next_t = t + gridDim.x;
        const bool has_next = (next_t < num_tiles);
        const int next_stage = stage ^ 1;

        // Prefetch next tile into other stage
        if (tid == 0 && has_next) {
            int nr = next_t / num_col_tiles, nc = next_t % num_col_tiles;
            tma::expect(q_arrived[next_stage], *bf16_stages[next_stage]);
            tma::load_async(*bf16_stages[next_stage], bf16_in, {nr, nc}, q_arrived[next_stage]);
        }

        // Wait for current tile
        __syncthreads();
        int &cur_phase = (stage == 0) ? phase0 : phase1;
        wait(q_arrived[stage], cur_phase);

        // Quantize current tile (128 active threads)
        auto &bf16_cur = *bf16_stages[stage];
        auto &fp4_cur  = *reinterpret_cast<typename G::Q_fp4_tile*>(&bf16_cur);
        typename G::Q_sc_vec (&sc_cur)[2] = *reinterpret_cast<typename G::Q_sc_vec(*)[2]>(
            reinterpret_cast<uint64_t>(&fp4_cur) + sizeof(fp4_cur));

        if (active) {
            bf16_2 regs[2][NKH][NPK];
            fp8e4m3 sc_reg[2][NKH];
            #pragma unroll
            for (int ch = 0; ch < 2; ch++) {
                #pragma unroll
                for (int i = 0; i < NKH; i++) {
                    const int kb = (i + tid/8)%NKH + ch*NKH;
                    #pragma unroll
                    for (int j = 0; j < NPK; j++) {
                        const int tc = kb*C::QK + ((tid+j)*2)%C::QK;
                        move<bf16_2>::lds(regs[ch][i][j],
                            static_cast<uint32_t>(__cvta_generic_to_shared(&bf16_cur)) +
                            (tile_row*C::QN + tc)*sizeof(bf16));
                    }
                }
            }
            #pragma unroll
            for (int ch = 0; ch < 2; ch++) {
                float amax[NKH];
                #pragma unroll
                for (int i = 0; i < NKH; i++) {
                    const int kb = (i + tid/8) % NKH;
                    bf16_2 mx = __habs2(regs[ch][i][0]);
                    #pragma unroll
                    for (int j = 1; j < NPK; j++)
                        mx = __hmax2(mx, __habs2(regs[ch][i][j]));
                    amax[kb] = __bfloat162float(__hmax(mx.x, mx.y));
                }
                #pragma unroll
                for (int i = 0; i < NKH; i++)
                    sc_reg[ch][i] = __nv_fp8_e4m3(amax[i] / 6.0f * S_ENC);
                #pragma unroll
                for (int i = 0; i < NKH; i++) {
                    const int kb = (i + tid/8) % NKH;
                    const float sld = static_cast<float>(sc_reg[ch][kb]);
                    const float se = 1.0f / fmaxf(sld * S_DEC, 1e-12f);
                    const int base = tile_row*C::QN/2 + (kb + ch*NKH)*C::QK/2;
                    #pragma unroll
                    for (int j = 0; j < NPK; j++) {
                        const int off = base + ((tid+j)&7);
                        const float2 sc = {
                            __bfloat162float(regs[ch][i][j].x) * se,
                            __bfloat162float(regs[ch][i][j].y) * se
                        };
                        asm volatile("{st.shared.b8 [%0], %1;}"
                            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&fp4_cur)) + off),
                               "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(sc, __NV_E2M1, cudaRoundNearest))));
                    }
                }
            }
            const int soff = (tile_row%32)*16 + (tile_row/32)*4;
            asm volatile("{st.shared.b32 [%0], %1;}"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sc_cur[0])) + soff),
                   "r"(*reinterpret_cast<uint32_t*>(&sc_reg[0][0])));
            asm volatile("{st.shared.b32 [%0], %1;}"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sc_cur[1])) + soff),
                   "r"(*reinterpret_cast<uint32_t*>(&sc_reg[1][0])));
        }

        __syncthreads();
        if (tid == 0) {
            tma::store_async(fp4_out, fp4_cur, {row, col});
            tma::store_async(sc_out, sc_cur[0], {row, col*2+0, 0});
            tma::store_async(sc_out, sc_cur[1], {row, col*2+1, 0});
            tma::store_commit_group();
            tma::store_async_wait<0>();
        }
        __syncthreads();
        cur_phase ^= 1;
        stage = next_stage;
    }
}

// ================================================================
// Phase 2: Standard FP4 GEMM
// ================================================================
template <typename C>
__device__ inline void gemm_phase(const globals<C> &g) {
    using G = globals<C>;
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
    const int num_red_blocks = 2 * g.A.cols() / C::Kb;
    const int num_blocks_per_sg = C::SUPERGROUP_SIZE * num_col_blocks;
    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_alloc((int*)&__shm[0]);
    typename G::input_tiles_t  (&itiles) [C::LOAD_PIPE_DEPTH] = sm_alloc.allocate<typename G::input_tiles_t, C::LOAD_PIPE_DEPTH>();
    typename G::input_scales_t (&iscales)[C::LOAD_PIPE_DEPTH] = sm_alloc.allocate<typename G::input_scales_t, C::LOAD_PIPE_DEPTH>();
    typename G::outputs_t      &otiles = sm_alloc.allocate<typename G::outputs_t>();
    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_alloc;

    __shared__ uint32_t tmem_addr;
    __shared__ semaphore tmem_prov;
    __shared__ semaphore tiles_arr[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore scales_arr[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_fin[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore out_arr;
    __shared__ semaphore out_fin;
    if (threadIdx.x == 32) {
        init_semaphore(tmem_prov, 0, 1);
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(tiles_arr[i], 0, 1);
            init_semaphore(scales_arr[i], 0, 1);
            init_semaphore(inputs_fin[i], 0, 1);
        }
        init_semaphore(out_arr, 0, 1);
        init_semaphore(out_fin, 0, C::CLUSTER_SIZE);
    }
    everyone::tma::cluster::arrive_aligned();
    const float gs = g.A_sc_global[{0}] * g.B_sc_global[{0}];

    if (warpgroup_id >= C::CONSUMER_WARPGROUPS && warp::elect_leader()) {
        int warp_id = group<WARPGROUP_WARPS*C::PRODUCER_WARPGROUPS>::warpid();
        if (warp_id == 3) {
            everyone::tma::cluster::wait();
            for (int bi = cluster_id; bi < num_blocks; bi += gridDim.x / C::CLUSTER_SIZE) {
                int sgi = bi / num_blocks_per_sg, iwsg = bi % num_blocks_per_sg;
                int rsg = min(C::SUPERGROUP_SIZE, num_row_blocks - sgi*C::SUPERGROUP_SIZE);
                int rbi = sgi*C::SUPERGROUP_SIZE + iwsg%rsg, cbi = iwsg/rsg;
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_fin[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(itiles[stage].A, g.A, {rbi*2+cta_id, i}, tiles_arr[stage], (uint16_t)(1<<cta_id), 0);
                    tma::cluster::load_async(itiles[stage].B, g.B, {cbi*2+cta_id, i}, tiles_arr[stage], (uint16_t)(1<<cta_id), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage+1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (warp_id == 2) {
            everyone::tma::cluster::wait();
            for (int bi = cluster_id; bi < num_blocks; bi += gridDim.x / C::CLUSTER_SIZE) {
                int sgi = bi / num_blocks_per_sg, iwsg = bi % num_blocks_per_sg;
                int rsg = min(C::SUPERGROUP_SIZE, num_row_blocks - sgi*C::SUPERGROUP_SIZE);
                int rbi = sgi*C::SUPERGROUP_SIZE + iwsg%rsg, cbi = iwsg/rsg;
                for (int i = 0; i < num_red_blocks; ++i) {
                    wait(inputs_fin[stage], get_phasebit<1>(phasebits, stage));
                    tma::cluster::load_async(iscales[stage].A, g.A_sc, {rbi*2+cta_id, i, 0}, scales_arr[stage], (uint16_t)(1<<cta_id), 0);
                    if constexpr (C::B_SC_SIZE==2) tma::cluster::load_async(iscales[stage].B[cta_id], g.B_sc, {cbi*2+cta_id, i, 0}, scales_arr[stage], (uint16_t)(0b11), 0);
                    else if (cta_id==0) tma::cluster::load_async(iscales[stage].B[0], g.B_sc, {cbi, i, 0}, scales_arr[stage], (uint16_t)(0b11), 0);
                    update_phasebit<1>(phasebits, stage);
                    stage = (stage+1) % C::LOAD_PIPE_DEPTH;
                }
            }
        } else if (cta_id == 0 && warp_id == 0) {
            everyone::tma::cluster::wait();
            wait(tmem_prov, 0);
            tm_alloc.set_addr(tmem_addr);
            auto out_tm  = tm_alloc.template allocate<full_tt_fl<C::Nb>>(0);
            auto A_sc_tm = tm_alloc.template allocate<full_tt_fp8e4m3<16*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256);
            auto B_sc_tm = tm_alloc.template allocate<full_tt_fp8e4m3<32*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH>>(256+4*C::MMA_PER_TILE*C::LOAD_PIPE_DEPTH);
            for (int bi = cluster_id; bi < num_blocks; bi += gridDim.x / C::CLUSTER_SIZE) {
                wait(out_fin, get_phasebit<1>(phasebits, 0));
                tensor_after_thread_sync();
                for (int i = 0; i < num_red_blocks; i++) {
                    tma::expect_bytes(scales_arr[stage], 2*sizeof(typename G::input_scales_t));
                    wait(scales_arr[stage], get_phasebit<0>(phasebits, stage));
                    #pragma unroll
                    for (int ii = 0; ii < C::MMA_PER_TILE; ii++) {
                        auto ast = A_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*16+ii*16);
                        auto &ass = *reinterpret_cast<st_fp8e4m3<32,16,false>*>(reinterpret_cast<uint64_t>(&iscales[stage].A.data[0])+16*32*ii);
                        load_mxnv_scale_async2(ast, ass);
                        auto bst0 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16);
                        auto &bss0 = *reinterpret_cast<st_fp8e4m3<32,16,false>*>(reinterpret_cast<uint64_t>(&iscales[stage].B[0].data[0])+16*32*ii);
                        load_mxnv_scale_async2(bst0, bss0);
                        if constexpr (C::B_SC_SIZE==2) {
                            auto bst1 = B_sc_tm.template subtile<full_tt_fp8e4m3<16>>(stage*C::MMA_PER_TILE*32+ii*C::B_SC_SIZE*16+16);
                            auto &bss1 = *reinterpret_cast<st_fp8e4m3<32,16,false>*>(reinterpret_cast<uint64_t>(&iscales[stage].B[1].data[0])+16*32*ii);
                            load_mxnv_scale_async2(bst1, bss1);
                        }
                    }
                    tma::expect_bytes(tiles_arr[stage], 2*sizeof(typename G::input_tiles_t));
                    wait(tiles_arr[stage], get_phasebit<0>(phasebits, stage));
                    if (i==0) mm2_ABt(out_tm, itiles[stage].A, itiles[stage].B,
                                      A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                      B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                      inputs_fin[stage]);
                    else      mma2_ABt(out_tm, itiles[stage].A, itiles[stage].B,
                                      A_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*16>>(stage*C::MMA_PER_TILE*16),
                                      B_sc_tm.template subtile<full_tt_fp8e4m3<C::MMA_PER_TILE*32>>(stage*C::MMA_PER_TILE*32),
                                      inputs_fin[stage]);
                    update_phasebit<0>(phasebits, stage);
                    stage = (stage+1) % C::LOAD_PIPE_DEPTH;
                }
                tensor_commit<2>(out_arr);
                update_phasebit<1>(phasebits, 0);
            }
        }
    } else if (warpgroup_id < C::CONSUMER_WARPGROUPS) {
        everyone::tma::cluster::wait_aligned();
        if (warpgroup::warpid() == 0) { tm_alloc.provision(tmem_addr); warp::arrive(tmem_prov); }
        wait(tmem_prov, 0);
        tm_alloc.set_addr(tmem_addr);
        auto out_tm = tm_alloc.template allocate<full_tt_fl<C::Nb>>(0);
        for (int bi = cluster_id; bi < num_blocks; bi += gridDim.x / C::CLUSTER_SIZE) {
            int sgi = bi / num_blocks_per_sg, iwsg = bi % num_blocks_per_sg;
            int rsg = min(C::SUPERGROUP_SIZE, num_row_blocks - sgi*C::SUPERGROUP_SIZE);
            int rbi = sgi*C::SUPERGROUP_SIZE + iwsg%rsg, cbi = iwsg/rsg;
            wait(out_arr, get_phasebit<0>(phasebits, 0));
            if constexpr (C::OVERLAP_EPI) {
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> Dr;
                    warpgroup::load_async(Dr, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    if (i == C::EPI_PIPE_DEPTH-1) { tensor_load_wait(); tensor_before_thread_sync(); warpgroup::sync(1); warpgroup::tma::cluster::arrive(out_fin, 0, 1); }
                    warp::mul(Dr, Dr, gs);
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>(); warpgroup::sync(1);
                    warpgroup::store(otiles.D[i%C::NUM_D_TILES], Dr); warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, otiles.D[i%C::NUM_D_TILES], {rbi*2+cta_id, C::EPI_PIPE_DEPTH*cbi+i});
                }
            } else {
                rt_bf<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> Dbf[C::EPI_PIPE_DEPTH];
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    rt_fl<C::Mb/8, C::Nb/C::EPI_PIPE_DEPTH> Dr;
                    warpgroup::load_async(Dr, out_tm.template subtile<full_tt_fl<C::Nb/C::EPI_PIPE_DEPTH>>(0, C::Nb/C::EPI_PIPE_DEPTH*i));
                    warp::mul(Dr, Dr, gs); warp::copy(Dbf[i], Dr);
                }
                tensor_load_wait(); tensor_before_thread_sync(); warpgroup::sync(1);
                warpgroup::tma::cluster::arrive(out_fin, 0, 1);
                #pragma unroll
                for (int i = 0; i < C::EPI_PIPE_DEPTH; i++) {
                    warpgroup::tma::store_async_read_wait<C::NUM_D_TILES-1>(); warpgroup::sync(1);
                    warpgroup::store(otiles.D[i%C::NUM_D_TILES], Dbf[i]); warpgroup::sync(1);
                    warpgroup::tma::store_async<dim::ROW, C::D_CACHE_POLICY>(g.D, otiles.D[i%C::NUM_D_TILES], {rbi*2+cta_id, C::EPI_PIPE_DEPTH*cbi+i});
                }
            }
            update_phasebit<0>(phasebits, 0);
        }
        warpgroup::sync(1); warpgroup::tma::store_async_read_wait<0>();
        if (warpgroup::warpid() == 0) tm_alloc.deprovision();
    }
}

template <typename C>
__device__ inline void kernel(const globals<C> &g) {
    using G = globals<C>;
    quantize_operand<C, G>(g.A_bf16, g.A_q_fp4, g.A_q_sc);
    if (g.quantize_b) quantize_operand<C, G>(g.B_bf16, g.B_q_fp4, g.B_q_sc);
    grid_barrier(g.barrier, gridDim.x);
    gemm_phase<C>(g);
}

} // namespace nvfp4_persistent_gemm
