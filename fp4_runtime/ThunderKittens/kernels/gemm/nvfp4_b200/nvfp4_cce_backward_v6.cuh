#pragma once

#include "nvfp4_cce_backward_v3.cuh"

namespace nvfp4_cce_backward_v6 {

#ifndef COMBO_DC_POST_MM2_SEM_DEBUG_MODE
#define COMBO_DC_POST_MM2_SEM_DEBUG_MODE 8
#endif

#ifndef COMBO_DC_POST_MM2_USE_SEM_OVERLOAD
#define COMBO_DC_POST_MM2_USE_SEM_OVERLOAD 1
#endif

#ifndef COMBO_DC_POST_MM2_ORDER_LEVEL
#define COMBO_DC_POST_MM2_ORDER_LEVEL 0
#endif

#ifndef COMBO_DC_STANDALONE_TRACE_TMEM
#define COMBO_DC_STANDALONE_TRACE_TMEM 1
#endif

#ifndef COMBO_DC_STANDALONE_TRY_DEPROVISION
#define COMBO_DC_STANDALONE_TRY_DEPROVISION 1
#endif

#ifndef COMBO_DC_STANDALONE_BSCALE_LOOP_MODE
#define COMBO_DC_STANDALONE_BSCALE_LOOP_MODE 0
#endif

#ifndef COMBO_DC_STANDALONE_BSCALE_DEST_MODE
#define COMBO_DC_STANDALONE_BSCALE_DEST_MODE 0
#endif

#ifndef COMBO_DC_STANDALONE_BSCALE_SOURCE_MODE
#define COMBO_DC_STANDALONE_BSCALE_SOURCE_MODE 0
#endif

#ifndef COMBO_DC_STANDALONE_STAGE
#define COMBO_DC_STANDALONE_STAGE 0
#endif

template <typename C>
using globals_5wg = nvfp4_cce_backward_v3::globals_5wg<C>;

template <typename C>
using globals_3wg = nvfp4_cce_backward_v3::globals_3wg<C>;

template <typename C, int SYNC_ID>
__device__ inline void load_combo_row_stage_from_global(
    typename globals_5wg<C>::combo_row_stage_t& stage,
    const globals_5wg<C>& g,
    int row_block_idx,
    int col_block_idx,
    int cta_id)
{
    const int lane_id = threadIdx.x % WARP_THREADS;
    const int row_fp4_stride = g.G_fp4_row.cols();
    const int global_row_base = row_block_idx * C::Mb + cta_id * (C::Mb / 2);
    const int global_fp4x2_col_base = col_block_idx * (C::Nb / 2);
    uint8_t* dst_fp4 = reinterpret_cast<uint8_t*>(&stage.G_row.data[0]);
    uint8_t* dst_sc = reinterpret_cast<uint8_t*>(&stage.G_row_sc.data[0]);
    const uint8_t* src_fp4 = reinterpret_cast<const uint8_t*>(g.G_fp4_row.raw_ptr);
    const uint8_t* src_sc = reinterpret_cast<const uint8_t*>(g.G_sc_row_ptr);

    for (int idx = warpgroup::warpid() * WARP_THREADS + lane_id;
         idx < static_cast<int>(sizeof(typename globals_5wg<C>::combo_row_stage_t));
         idx += WARPGROUP_WARPS * WARP_THREADS) {
        dst_fp4[idx] = 0;
    }
    for (int local_row = warpgroup::warpid(); local_row < C::Mb / 2; local_row += WARPGROUP_WARPS) {
        const int global_row = global_row_base + local_row;
        const int fp4_offset = global_row * row_fp4_stride + global_fp4x2_col_base;
        const int local_offset = local_row * globals_5wg<C>::G_fp4_row_tile::cols;
        for (int fp4x2_col = lane_id; fp4x2_col < globals_5wg<C>::G_fp4_row_tile::cols; fp4x2_col += WARP_THREADS) {
            dst_fp4[local_offset + fp4x2_col] = src_fp4[fp4_offset + fp4x2_col];
        }
        const int depth = global_row / 128;
        const int sr = global_row % 32;
        const int rr = (global_row / 32) % 4;
        for (int local_col16_idx = lane_id; local_col16_idx < C::Nb / 16; local_col16_idx += WARP_THREADS) {
            const int global_col16 = col_block_idx * (C::Nb / 16) * 16 + local_col16_idx * 16;
            const int global_kgroup = global_col16 / 64;
            const int col16_in_64 = (global_col16 / 16) % 4;
            const int chunk = depth * g.G_sc_row_kgroups + global_kgroup;
            const int byte_idx = sr * 16 + rr * 4 + col16_in_64;
            const int local_kgroup = local_col16_idx / 4;
            const int local_byte_idx = (local_row % 32) * 16 + ((local_row / 32) % 4) * 4 + col16_in_64;
            dst_sc[local_kgroup * 512 + local_byte_idx] = src_sc[chunk * 512 + byte_idx];
        }
    }
    return;
}

template <typename C, int SYNC_ID>
__device__ inline void load_combo_col_stage_from_global(
    typename globals_5wg<C>::combo_col_stage_t& stage,
    const globals_5wg<C>& g,
    int row_block_idx,
    int col_block_idx)
{
    const int lane_id = threadIdx.x % WARP_THREADS;
    const int col_fp4_stride = g.A.rows() / 2;
    const int global_col_base = col_block_idx * C::Nb;
    const int global_row_pair_base = row_block_idx * (C::Mb / 2);
    uint8_t* dst_fp4 = reinterpret_cast<uint8_t*>(&stage.Gt_row.data[0]);
    uint8_t* dst_sc = reinterpret_cast<uint8_t*>(&stage.Gt_row_sc.data[0]);
    const uint8_t* src_fp4 = reinterpret_cast<const uint8_t*>(g.G_fp4_col_ptr);
    const uint8_t* src_sc = reinterpret_cast<const uint8_t*>(g.G_sc_col_ptr);

    for (int idx = warpgroup::warpid() * WARP_THREADS + lane_id;
         idx < static_cast<int>(sizeof(typename globals_5wg<C>::combo_col_stage_t));
         idx += WARPGROUP_WARPS * WARP_THREADS) {
        dst_fp4[idx] = 0;
    }
    warpgroup::sync(SYNC_ID);

    for (int local_col = warpgroup::warpid(); local_col < C::Nb; local_col += WARPGROUP_WARPS) {
        const int global_col = global_col_base + local_col;
        const int src_offset = global_col * col_fp4_stride + global_row_pair_base;
        const int dst_offset = local_col * globals_5wg<C>::combo_col_fp4_tile::cols;
        for (int row_pair = lane_id; row_pair < globals_5wg<C>::combo_col_fp4_tile::cols; row_pair += WARP_THREADS) {
            dst_fp4[dst_offset + row_pair] = src_fp4[src_offset + row_pair];
        }
        const int sr = global_col % 32;
        const int rr = (global_col / 32) % 4;
        for (int local_row16 = lane_id; local_row16 < C::Mb / 16; local_row16 += WARP_THREADS) {
            const int global_row_base16 = row_block_idx * C::Mb + local_row16 * 16;
            const int m_kgroup = global_row_base16 / 64;
            const int m_16_in_64 = (global_row_base16 / 16) % 4;
            const int byte_idx = sr * 16 + rr * 4 + m_16_in_64;
            dst_sc[m_kgroup * 512 + byte_idx] = src_sc[m_kgroup * 512 + byte_idx];
        }
    }
    warpgroup::sync(SYNC_ID);
}

template <typename C>
__device__ inline void issue_combo_de_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    uint32_t &combo_de_tmem_addr,
    semaphore &combo_de_tmem_provisioned,
    semaphore &combo_de_p3_c_tiles_arrived,
    semaphore &combo_de_p3_c_scales_arrived,
    semaphore &combo_de_p3_inputs_finished,
    semaphore (&combo_de_p3_outputs_arrived)[2],
    semaphore (&combo_de_p3_outputs_finished)[2])
{
    const bool combo_issue_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);
    (void)cta_id;
    (void)cluster_id;
    (void)combo_de_tmem_addr;
    (void)combo_de_tmem_provisioned;
    (void)combo_de_p3_c_tiles_arrived;
    (void)combo_de_p3_c_scales_arrived;
    (void)combo_de_p3_outputs_arrived;
    (void)combo_de_p3_outputs_finished;
    if (combo_issue_leader) {
        g.G_sg_row[{0}] = 271.0f;
    }
    return;
}

template <typename C>
__device__ inline void drain_combo_de_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    uint32_t &combo_de_tmem_addr,
    semaphore &combo_de_tmem_provisioned,
    semaphore &combo_de_p3_inputs_finished,
    semaphore (&combo_de_p3_outputs_arrived)[2],
    semaphore (&combo_de_p3_outputs_finished)[2])
{
    (void)combo_de_tmem_addr;
    (void)combo_de_tmem_provisioned;
    (void)combo_de_p3_inputs_finished;
    (void)combo_de_p3_outputs_arrived;
    (void)combo_de_p3_outputs_finished;
    (void)cta_id;
    (void)cluster_id;
    if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
        g.G_sg_row[{0}] = 273.0f;
    }
    return;
}

template <typename C>
__device__ inline void issue_combo_dc_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    volatile int* combo_dc_debug_mailbox,
    semaphore *combo_dc_debug_sem,
    uint32_t &combo_dc_tmem_addr,
    semaphore &combo_dc_tmem_provisioned,
    semaphore &combo_dc_p3_e_tiles_arrived,
    semaphore &combo_dc_p3_e_scales_arrived,
    semaphore &combo_dc_p3_inputs_finished,
    semaphore (&combo_dc_p3_outputs_arrived)[3],
    semaphore (&combo_dc_p3_outputs_committed)[3],
    semaphore (&combo_dc_p3_outputs_finished)[3])
{
    using G = globals_5wg<C>;
    constexpr int combo_dc_a_scale_chunks = C::Nb / 64;
    constexpr int combo_dc_b_scale_chunks = C::Nb / 64;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    auto &combo_col_stage = sm_allocator.template allocate<typename G::combo_col_stage_t>();
    auto &combo_dc_tile_stage = sm_allocator.template allocate<typename G::combo_dc_tile_stage_t>();
    auto &combo_dc_scales_stage = sm_allocator.template allocate<typename G::combo_dc_scales_stage_t>();
    auto &combo_dc_b_sc_src_stage = sm_allocator.template allocate<st_fp8e4m3<32, 16, false>>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    const bool combo_issue_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);
    constexpr bool combo_dc_standalone_trace_tmem = COMBO_DC_STANDALONE_TRACE_TMEM != 0;
    constexpr int combo_dc_standalone_stage = COMBO_DC_STANDALONE_STAGE;
    constexpr bool combo_dc_standalone_trace_verbose =
        combo_dc_standalone_trace_tmem && combo_dc_standalone_stage == 0;
    constexpr int combo_dc_standalone_bscale_loop_mode =
        COMBO_DC_STANDALONE_BSCALE_LOOP_MODE;
    constexpr int combo_dc_standalone_bscale_dest_mode =
        COMBO_DC_STANDALONE_BSCALE_DEST_MODE;
    constexpr int combo_dc_standalone_bscale_source_mode =
        COMBO_DC_STANDALONE_BSCALE_SOURCE_MODE;
    const int combo_issue_local_idx = warpgroup::warpid() * WARP_THREADS + warp::laneid();

    if (warpgroup::warpid() == 0) {
        tm_allocator.provision(combo_dc_tmem_addr);
        warp::arrive(combo_dc_tmem_provisioned);
    }
    wait(combo_dc_tmem_provisioned, 0);
    if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
        cluster_id == 0 && cta_id == 0) {
        printf("combo_dc tmem provision addr=%u standalone=%d\n",
               combo_dc_tmem_addr, combo_dc_debug_mailbox != nullptr ? 1 : 0);
    }
    tm_allocator.set_addr(combo_dc_tmem_addr);

    auto combo_dc_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
    auto combo_dc_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(C::Nb);
    auto combo_dc_tm_2 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(2 * C::Nb);
    auto combo_dc_a_sc_tm =
        tm_allocator.template allocate<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(3 * C::Nb);
    auto combo_dc_b_sc_tm =
        tm_allocator.template allocate<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(
            3 * C::Nb + 4 * combo_dc_a_scale_chunks);

    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_col_blocks = g.B.rows() / C::Nb;
    const int num_blocks = num_row_blocks * num_col_blocks;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;

    int combo_dc_e_tiles_phase = 0;
    int combo_dc_e_scales_phase = 0;
    int combo_dc_inputs_phase = 1;
    int combo_dc_outputs_finished_phase[3] = {0, 0, 0};
    bool first_input_issue = true;
    bool first_output_use[3] = {true, true, true};

    #pragma unroll 1
    for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / C::CLUSTER_SIZE) {
        const int supergroup_idx = block_idx / num_blocks_per_supergroup;
        const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
        const int rows_in_supergroup =
            min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
        const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
        const int row_block_idx = supergroup_idx * C::SUPERGROUP_SIZE + row_within_supergroup;
        const int col_block_idx = idx_within_supergroup / rows_in_supergroup;

        load_combo_col_stage_from_global<C, 4>(combo_col_stage, g, row_block_idx, col_block_idx);
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        warpgroup::sync(4);

        #pragma unroll
        for (int ii = 0; ii < combo_dc_a_scale_chunks; ++ii) {
            auto combo_a_sc_sub =
                combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
            auto &combo_gt_sc_sm_sub =
                *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                    reinterpret_cast<uint64_t>(&combo_col_stage.Gt_row_sc.data[0]) + 16 * 32 * ii);
            load_mxnv_scale_async2(combo_a_sc_sub, combo_gt_sc_sm_sub);
        }

        const int num_k_blocks = g.dC_out.cols() / C::Nb;
        const int debug_num_k_blocks =
            min(num_k_blocks, combo_dc_debug_mailbox != nullptr ? 1 : 4);
        constexpr int combo_dc_post_mm2_sem_debug_mode = COMBO_DC_POST_MM2_SEM_DEBUG_MODE;
        constexpr bool combo_dc_post_mm2_use_sem_overload =
            COMBO_DC_POST_MM2_USE_SEM_OVERLOAD != 0;
        constexpr int combo_dc_post_mm2_order_level = COMBO_DC_POST_MM2_ORDER_LEVEL;
        constexpr bool combo_dc_slot1_skip_real_inputs_wait = true;
        constexpr int combo_dc_slot2_debug_cut = 14;
        constexpr bool combo_dc_debug_compare_slot1_normal_publish = true;
        constexpr bool combo_dc_debug_slot1_direct_arrived_commit = true;
        constexpr bool combo_dc_slot2_skip_real_inputs_wait =
            !combo_dc_debug_compare_slot1_normal_publish;
        constexpr int combo_dc_slot3_debug_cut = 6;
        constexpr int combo_dc_slot4_debug_cut = 10;
#define COMBO_DC_STAGE_RETURN_ISSUE(STAGE_VALUE, LABEL)                                      \
        do {                                                                                  \
            if (combo_dc_standalone_stage == (STAGE_VALUE) &&                                 \
                combo_dc_debug_mailbox != nullptr && combo_issue_leader &&                    \
                cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {                        \
                printf("combo_dc stage %d %s addr=%u\n",                                      \
                       (STAGE_VALUE), LABEL, combo_dc_tmem_addr);                             \
                return;                                                                       \
            }                                                                                 \
        } while (0)
        #pragma unroll 1
        for (int k_block_idx = 0; k_block_idx < debug_num_k_blocks; ++k_block_idx) {
            const int slot = (k_block_idx < 3) ? k_block_idx : (k_block_idx & 1);
            auto combo_dc_tm =
                (slot == 0) ? combo_dc_tm_0 : ((slot == 1) ? combo_dc_tm_1 : combo_dc_tm_2);

            if (combo_issue_leader && combo_dc_debug_mailbox != nullptr && k_block_idx == 2) {
                combo_dc_debug_mailbox[2] = 2;
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            }

            if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_post_mm2_sem_debug_mode == -4) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 376.0f;
                }
                return;
            }
            if (combo_issue_leader) {
                if (!first_input_issue) {
                    if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr &&
                        combo_dc_slot1_skip_real_inputs_wait) {
                    } else if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                        combo_dc_slot2_skip_real_inputs_wait) {
                        g.G_sg_row[{0}] = 400.0f;
                    } else if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr) {
                        while (combo_dc_debug_mailbox[0] == 0) {}
                        combo_dc_debug_mailbox[0] = 0;
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        g.G_sg_row[{0}] = 420.0f;
                    } else {
                        wait(combo_dc_p3_inputs_finished, combo_dc_inputs_phase);
                        combo_dc_inputs_phase ^= 1;
                    }
                }
                if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                    combo_dc_slot4_debug_cut == 0) {
                    g.G_sg_row[{0}] = 440.0f;
                    return;
                }
                if (!first_output_use[slot]) {
                    wait(combo_dc_p3_outputs_finished[slot], combo_dc_outputs_finished_phase[slot]);
                    tensor_after_thread_sync();
                    combo_dc_outputs_finished_phase[slot] ^= 1;
                }
                if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                    combo_dc_slot4_debug_cut == 1) {
                    g.G_sg_row[{0}] = 441.0f;
                    return;
                }
                tma::expect_bytes(combo_dc_p3_e_tiles_arrived, sizeof(typename G::combo_p3_E_tile));
                tma::load_async(
                    combo_dc_tile_stage.E_operand, g.E_col,
                    {k_block_idx * 2 + cta_id, row_block_idx},
                    combo_dc_p3_e_tiles_arrived);
                tma::expect_bytes(combo_dc_p3_e_scales_arrived, sizeof(typename G::combo_p3_E_sc_tile));
                tma::load_async(
                    combo_dc_scales_stage.E_sc, g.E_col_sc,
                    {k_block_idx, row_block_idx, 0},
                    combo_dc_p3_e_scales_arrived);
            }
            warpgroup::sync(4);
            if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                printf("combo_dc slot0 issued_tma addr=%u\n", combo_dc_tmem_addr);
            }
            COMBO_DC_STAGE_RETURN_ISSUE(1, "producer_after_tma_issue");
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_debug_cut == 2) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 442.0f;
                }
                return;
            }
            if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_post_mm2_sem_debug_mode == -3) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 377.0f;
                }
                return;
            }
            if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_debug_compare_slot1_normal_publish) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 399.0f;
                }
                return;
            }

            if (combo_issue_leader) {
                wait(combo_dc_p3_e_scales_arrived, combo_dc_e_scales_phase);
                combo_dc_e_scales_phase ^= 1;
            }
            warpgroup::sync(4);
            if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                printf("combo_dc slot0 saw_e_scales addr=%u\n", combo_dc_tmem_addr);
            }
            COMBO_DC_STAGE_RETURN_ISSUE(2, "producer_after_e_scales_wait");
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_debug_cut == 3) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 443.0f;
                }
                return;
            }
            if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot2_debug_cut == 0) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 402.0f;
                }
                return;
            }
            if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot3_debug_cut == 0) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 421.0f;
                }
                return;
            }
            if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_post_mm2_sem_debug_mode == -2) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 378.0f;
                }
                return;
            }

            if constexpr (combo_dc_standalone_bscale_loop_mode == 2) {
                if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                    cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                    printf("combo_dc slot0 skipped_b_scales loop=%d dest=%d source=%d addr=%u\n",
                           combo_dc_standalone_bscale_loop_mode,
                           combo_dc_standalone_bscale_dest_mode,
                           combo_dc_standalone_bscale_source_mode,
                           combo_dc_tmem_addr);
                }
            } else {
                #pragma unroll
                for (int ii = 0; ii < combo_dc_b_scale_chunks; ++ii) {
                    if constexpr (combo_dc_standalone_bscale_loop_mode == 1) {
                        if (ii > 0) continue;
                    }
                    auto combo_b_sc_sub =
                        combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<16>>(ii * 16);
                    auto combo_a_sc_sub =
                        combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16>>(0);
                    auto &combo_e_sc_sm_sub =
                        *reinterpret_cast<st_fp8e4m3<32, 16, false> *>(
                            reinterpret_cast<uint64_t>(&combo_dc_scales_stage.E_sc.data[0]) + 16 * 32 * ii);

                    if constexpr (combo_dc_standalone_bscale_source_mode == 1) {
                        if (ii == 0) {
                            uint8_t *dst_bytes = reinterpret_cast<uint8_t*>(&combo_dc_b_sc_src_stage.data[0]);
                            const uint8_t *src_bytes = reinterpret_cast<const uint8_t*>(&combo_e_sc_sm_sub.data[0]);
                            for (int idx = combo_issue_local_idx;
                                 idx < static_cast<int>(sizeof(st_fp8e4m3<32, 16, false>));
                                 idx += WARPGROUP_WARPS * WARP_THREADS) {
                                dst_bytes[idx] = src_bytes[idx];
                            }
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            warpgroup::sync(4);
                        }
                    }

                    if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                        cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                        printf("combo_dc slot0 b_sc_before ii=%d loop=%d dest=%d source=%d addr=%u\n",
                               ii,
                               combo_dc_standalone_bscale_loop_mode,
                               combo_dc_standalone_bscale_dest_mode,
                               combo_dc_standalone_bscale_source_mode,
                               combo_dc_tmem_addr);
                    }

                    if constexpr (combo_dc_standalone_bscale_dest_mode == 1) {
                        if constexpr (combo_dc_standalone_bscale_source_mode == 1) {
                            if (ii == 0) {
                                load_mxnv_scale_async2(combo_a_sc_sub, combo_dc_b_sc_src_stage);
                            } else {
                                load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
                            }
                        } else {
                            if (ii == 0) {
                                load_mxnv_scale_async2(combo_a_sc_sub, combo_e_sc_sm_sub);
                            } else {
                                load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
                            }
                        }
                    } else {
                        if constexpr (combo_dc_standalone_bscale_source_mode == 1) {
                            if (ii == 0) {
                                load_mxnv_scale_async2(combo_b_sc_sub, combo_dc_b_sc_src_stage);
                            } else {
                                load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
                            }
                        } else {
                            load_mxnv_scale_async2(combo_b_sc_sub, combo_e_sc_sm_sub);
                        }
                    }

                    if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                        cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                        printf("combo_dc slot0 b_sc_after ii=%d loop=%d dest=%d source=%d addr=%u\n",
                               ii,
                               combo_dc_standalone_bscale_loop_mode,
                               combo_dc_standalone_bscale_dest_mode,
                               combo_dc_standalone_bscale_source_mode,
                               combo_dc_tmem_addr);
                    }
                }
            }
            if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                printf("combo_dc slot0 loaded_b_scales addr=%u\n", combo_dc_tmem_addr);
            }
            COMBO_DC_STAGE_RETURN_ISSUE(3, "producer_after_b_scale_loop");

            if (combo_issue_leader) {
                wait(combo_dc_p3_e_tiles_arrived, combo_dc_e_tiles_phase);
                combo_dc_e_tiles_phase ^= 1;
            }
            if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                printf("combo_dc slot0 saw_e_tiles addr=%u\n", combo_dc_tmem_addr);
            }
            COMBO_DC_STAGE_RETURN_ISSUE(4, "producer_after_e_tiles_wait");
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_debug_cut == 4) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 444.0f;
                }
                return;
            }
            if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot2_debug_cut == 1) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 403.0f;
                }
                return;
            }
            if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot3_debug_cut == 1) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 422.0f;
                }
                return;
            }
            if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_post_mm2_sem_debug_mode == -1) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 379.0f;
                }
                return;
            }
            if constexpr (combo_dc_post_mm2_use_sem_overload) {
                mm2_ABt(
                    combo_dc_tm, combo_col_stage.Gt_row, combo_dc_tile_stage.E_operand,
                    combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(0),
                    combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(0),
                    combo_dc_p3_inputs_finished);
            } else {
                mm2_ABt(
                    combo_dc_tm, combo_col_stage.Gt_row, combo_dc_tile_stage.E_operand,
                    combo_dc_a_sc_tm.template subtile<full_tt_fp8e4m3<16 * combo_dc_a_scale_chunks>>(0),
                    combo_dc_b_sc_tm.template subtile<full_tt_fp8e4m3<32 * combo_dc_b_scale_chunks>>(0));
            }
            if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                printf("combo_dc slot0 mm2_done addr=%u\n", combo_dc_tmem_addr);
            }
            COMBO_DC_STAGE_RETURN_ISSUE(5, "producer_after_mm2");
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_debug_cut == 5) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 445.0f;
                }
                return;
            }
            if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr) {
                if constexpr (combo_dc_post_mm2_sem_debug_mode == 6 ||
                              combo_dc_post_mm2_sem_debug_mode == 7 ||
                              combo_dc_post_mm2_sem_debug_mode == 8) {
                    if constexpr (combo_dc_post_mm2_order_level >= 1) {
                        tensor_after_thread_sync();
                    }
                    if constexpr (combo_dc_post_mm2_order_level >= 2) {
                        asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                    }
                    if constexpr (combo_dc_post_mm2_order_level >= 3) {
                        warpgroup::sync(4);
                    }
                }
                if (combo_issue_leader) {
                    if constexpr (combo_dc_post_mm2_sem_debug_mode == 1) {
                        combo_dc_debug_mailbox[1] = 1;
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                        g.G_sg_row[{0}] = 381.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 2) {
                        arrive(*combo_dc_debug_sem);
                        g.G_sg_row[{0}] = 382.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 3) {
                        arrive(*combo_dc_debug_sem);
                        wait(*combo_dc_debug_sem, 0);
                        g.G_sg_row[{0}] = 383.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 4) {
                        arrive(combo_dc_p3_outputs_committed[2]);
                        g.G_sg_row[{0}] = 384.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 5) {
                        arrive(combo_dc_p3_outputs_arrived[2]);
                        g.G_sg_row[{0}] = 385.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 6) {
                        wait(combo_dc_p3_inputs_finished, 1);
                        g.G_sg_row[{0}] = 386.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 7) {
                        arrive(*combo_dc_debug_sem);
                        wait(*combo_dc_debug_sem, 0);
                        g.G_sg_row[{0}] = 387.0f;
                    } else if constexpr (combo_dc_post_mm2_sem_debug_mode == 8) {
                        g.G_sg_row[{0}] = 388.0f;
                    } else {
                        g.G_sg_row[{0}] = 380.0f;
                    }
                }
                if constexpr (combo_dc_post_mm2_sem_debug_mode != 8) {
                    return;
                }
            }
            if (k_block_idx == 2) {
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot2_debug_cut == 2) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 404.0f;
                    }
                    return;
                }
                tensor_commit<2>(combo_dc_p3_outputs_committed[slot]);
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot2_debug_cut == 3) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 405.0f;
                    }
                    return;
                }
                tensor_after_thread_sync();
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot2_debug_cut == 4) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 406.0f;
                    }
                    return;
                }
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot2_debug_cut == 5) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 407.0f;
                    }
                    return;
                }
                if (combo_issue_leader) {
                    arrive(combo_dc_p3_outputs_arrived[slot]);
                    if (combo_dc_debug_mailbox != nullptr) {
                        combo_dc_debug_mailbox[2] = 1;
                        __threadfence_block();
                        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    }
                }
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot2_debug_cut == 6) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 408.0f;
                    }
                    return;
                }
                warpgroup::sync(4);
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot2_debug_cut == 7) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 409.0f;
                    }
                    return;
                }
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 395.0f;
                }
                if constexpr (combo_dc_slot2_debug_cut == 14) {
                    continue;
                }
                return;
            }
            if (k_block_idx == 3) {
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot3_debug_cut == 2) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 423.0f;
                    }
                    return;
                }
                tensor_commit<2>(combo_dc_p3_outputs_committed[slot]);
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot3_debug_cut == 3) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 424.0f;
                    }
                    return;
                }
                tensor_after_thread_sync();
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (combo_issue_leader) {
                    arrive(combo_dc_p3_outputs_arrived[slot]);
                }
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot3_debug_cut == 4) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 425.0f;
                    }
                    return;
                }
                warpgroup::sync(4);
                if (combo_dc_debug_mailbox != nullptr && combo_dc_slot3_debug_cut == 5) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 426.0f;
                    }
                    return;
                }
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 427.0f;
                }
                continue;
            }
            if (k_block_idx == 1) {
                if (combo_dc_debug_mailbox != nullptr) {
                    if (combo_dc_debug_compare_slot1_normal_publish) {
                        if (combo_dc_debug_slot1_direct_arrived_commit) {
                            tensor_commit<2>(combo_dc_p3_outputs_arrived[slot]);
                            tensor_after_thread_sync();
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                            warpgroup::sync(4);
                            if (combo_issue_leader) {
                                combo_dc_debug_mailbox[1] = 3;
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                g.G_sg_row[{0}] = 387.0f;
                            }
                        } else {
                            tensor_commit<2>(combo_dc_p3_outputs_committed[slot]);
                            tensor_after_thread_sync();
                            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                            if (combo_issue_leader) {
                                arrive(combo_dc_p3_outputs_arrived[slot]);
                                combo_dc_debug_mailbox[1] = 2;
                                __threadfence_block();
                                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                                g.G_sg_row[{0}] = 386.0f;
                            }
                            warpgroup::sync(4);
                        }
                    } else {
                        tensor_commit<2>(combo_dc_p3_outputs_committed[slot]);
                        if (combo_issue_leader) {
                            combo_dc_debug_mailbox[1] = 1;
                            __threadfence_block();
                            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                            g.G_sg_row[{0}] = 384.0f;
                        }
                    }
                    first_input_issue = false;
                    first_output_use[slot] = false;
                    continue;
                }
                tensor_commit<2>(combo_dc_p3_outputs_committed[slot]);
                tensor_after_thread_sync();
                asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
                if (combo_issue_leader) {
                    arrive(combo_dc_p3_outputs_arrived[slot]);
                }
                warpgroup::sync(4);
                first_input_issue = false;
                first_output_use[slot] = false;
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 385.0f;
                }
                continue;
            }
            tensor_commit<2>(combo_dc_p3_outputs_committed[slot]);
            COMBO_DC_STAGE_RETURN_ISSUE(6, "producer_after_tensor_commit");
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_debug_cut == 6) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 446.0f;
                }
                return;
            }
            tensor_after_thread_sync();
            COMBO_DC_STAGE_RETURN_ISSUE(7, "producer_after_tensor_after_thread_sync");
            asm volatile("fence.proxy.async.shared::cluster;\n" ::: "memory");
            COMBO_DC_STAGE_RETURN_ISSUE(8, "producer_after_cluster_fence");
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_debug_cut == 7) {
                if (combo_issue_leader) {
                    g.G_sg_row[{0}] = 447.0f;
                }
                return;
            }
            warpgroup::sync(4);
            if (combo_issue_leader) {
                arrive(combo_dc_p3_outputs_arrived[slot]);
            }
            COMBO_DC_STAGE_RETURN_ISSUE(9, "producer_after_outputs_arrived");
                if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                    combo_dc_slot4_debug_cut == 8) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 448.0f;
                    }
                    return;
                }
                warpgroup::sync(4);
                if (combo_dc_standalone_trace_verbose && combo_issue_leader &&
                    cluster_id == 0 && cta_id == 0 && k_block_idx == 0) {
                    printf("combo_dc slot0 published addr=%u\n", combo_dc_tmem_addr);
                }
                COMBO_DC_STAGE_RETURN_ISSUE(10, "producer_after_publish_sync");
                if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                    combo_dc_slot4_debug_cut == 9) {
                    if (combo_issue_leader) {
                        g.G_sg_row[{0}] = 449.0f;
                    }
                return;
            }
            first_input_issue = false;
            first_output_use[slot] = false;
        }
        if (combo_issue_leader) {
            g.G_sg_row[{0}] = 375.0f;
        }
    }
#undef COMBO_DC_STAGE_RETURN_ISSUE
}

template <typename C>
__device__ inline void drain_combo_dc_from_global_bridge(
    const globals_5wg<C>& g,
    int cta_id,
    int cluster_id,
    volatile int* combo_dc_debug_mailbox,
    uint32_t &combo_dc_tmem_addr,
    semaphore &combo_dc_tmem_provisioned,
    semaphore (&combo_dc_p3_outputs_arrived)[3],
    semaphore (&combo_dc_p3_outputs_finished)[3])
{
    using G = globals_5wg<C>;
    using combo_dc_rt = rt_fl<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;
    using combo_dc_rt_bf = rt_bf<C::Nb / 4, C::Nb / C::EPI_PIPE_DEPTH>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    auto &combo_col_stage = sm_allocator.template allocate<typename G::combo_col_stage_t>();
    auto &combo_dc_tile_stage = sm_allocator.template allocate<typename G::combo_dc_tile_stage_t>();
    auto &combo_dc_scales_stage = sm_allocator.template allocate<typename G::combo_dc_scales_stage_t>();
    auto &combo_dc_output_stage = sm_allocator.template allocate<typename G::combo_dc_output_stage_t>();

    tensor_allocator<1, C::CLUSTER_SIZE, false> tm_allocator;
    const bool combo_drain_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);
    constexpr bool combo_dc_standalone_trace_tmem = COMBO_DC_STANDALONE_TRACE_TMEM != 0;
    constexpr int combo_dc_standalone_stage = COMBO_DC_STANDALONE_STAGE;
    constexpr bool combo_dc_standalone_trace_verbose =
        combo_dc_standalone_trace_tmem && combo_dc_standalone_stage == 0;
    constexpr bool combo_dc_standalone_try_deprovision =
        COMBO_DC_STANDALONE_TRY_DEPROVISION != 0;
#define COMBO_DC_STAGE_RETURN_DRAIN(STAGE_VALUE, LABEL)                                      \
    do {                                                                                      \
        if (combo_dc_standalone_stage == (STAGE_VALUE) &&                                     \
            combo_dc_debug_mailbox != nullptr && combo_drain_leader &&                        \
            cluster_id == 0 && cta_id == 0) {                                                 \
            printf("combo_dc stage %d %s addr=%u\n",                                          \
                   (STAGE_VALUE), LABEL, combo_dc_tmem_addr);                                 \
            return;                                                                           \
        }                                                                                     \
    } while (0)
    (void)combo_col_stage;
    (void)combo_dc_tile_stage;
    (void)combo_dc_scales_stage;
    if (combo_dc_standalone_trace_verbose && combo_drain_leader &&
        cluster_id == 0 && cta_id == 0 && combo_dc_debug_mailbox != nullptr) {
        printf("combo_dc drain start addr=%u\n", combo_dc_tmem_addr);
    }
    COMBO_DC_STAGE_RETURN_DRAIN(11, "drain_entry");
    if (combo_drain_leader) {
        wait(combo_dc_p3_outputs_arrived[0], 0);
    }
    if (combo_dc_standalone_trace_verbose && combo_drain_leader &&
        cluster_id == 0 && cta_id == 0 && combo_dc_debug_mailbox != nullptr) {
        printf("combo_dc drain saw outputs_arrived0 addr=%u\n", combo_dc_tmem_addr);
    }
    COMBO_DC_STAGE_RETURN_DRAIN(12, "drain_after_outputs_arrived_wait");
    wait(combo_dc_tmem_provisioned, 0);
    if (combo_dc_standalone_trace_verbose && combo_drain_leader &&
        cluster_id == 0 && cta_id == 0 && combo_dc_debug_mailbox != nullptr) {
        printf("combo_dc drain saw tmem_provisioned addr=%u\n", combo_dc_tmem_addr);
    }
    COMBO_DC_STAGE_RETURN_DRAIN(13, "drain_after_tmem_provisioned_wait");
    tm_allocator.set_addr(combo_dc_tmem_addr);
    if (combo_dc_standalone_trace_verbose && combo_drain_leader &&
        cluster_id == 0 && cta_id == 0) {
        printf("combo_dc tmem drain sees addr=%u standalone=%d\n",
               combo_dc_tmem_addr, combo_dc_debug_mailbox != nullptr ? 1 : 0);
    }
    auto combo_dc_tm_0 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
    auto combo_dc_tm_1 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(C::Nb);
    auto combo_dc_tm_2 = tm_allocator.template allocate<full_tt_fl<C::Nb>>(2 * C::Nb);
    const int num_row_blocks = g.A.rows() / C::Mb;
    const int num_col_blocks = g.B.rows() / C::Nb;
    const int num_blocks_per_supergroup = C::SUPERGROUP_SIZE * num_col_blocks;
    const int block_idx = cluster_id;
    const int supergroup_idx = block_idx / num_blocks_per_supergroup;
    const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
    const int rows_in_supergroup =
        min(C::SUPERGROUP_SIZE, num_row_blocks - supergroup_idx * C::SUPERGROUP_SIZE);
    const int col_block_idx = idx_within_supergroup / rows_in_supergroup;
    const float combo_dc_scale = g.G_sg_row[{0}] * g.E_col_sc_global[{0}];
    const int num_k_blocks = g.dC_out.cols() / C::Nb;
        const int debug_num_k_blocks =
            min(num_k_blocks, combo_dc_debug_mailbox != nullptr ? 1 : 4);
    int combo_dc_outputs_arrived_phase[3] = {0, 0, 0};
        constexpr int combo_dc_slot2_debug_cut = 14;
        constexpr bool combo_dc_debug_neutralize_slot1_drain = false;
        constexpr int combo_dc_slot3_drain_debug_cut = 5;
        constexpr int combo_dc_slot4_drain_debug_cut = 5;

    #pragma unroll 1
    for (int k_block_idx = 0; k_block_idx < debug_num_k_blocks; ++k_block_idx) {
        const int slot = (k_block_idx < 3) ? k_block_idx : (k_block_idx & 1);
        auto combo_dc_tm =
            (slot == 0) ? combo_dc_tm_0 : ((slot == 1) ? combo_dc_tm_1 : combo_dc_tm_2);

        if (k_block_idx == 1 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_debug_neutralize_slot1_drain) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 402.0f;
            }
            continue;
        }
        if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot2_debug_cut < 8) {
            if (combo_drain_leader) {
                while (combo_dc_debug_mailbox[2] == 0) {}
                g.G_sg_row[{0}] = 401.0f;
            }
            return;
        }
        if (combo_drain_leader) {
            wait(combo_dc_p3_outputs_arrived[slot], combo_dc_outputs_arrived_phase[slot]);
            combo_dc_outputs_arrived_phase[slot] ^= 1;
        }
        if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot4_drain_debug_cut == 0) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 450.0f;
            }
            return;
        }
        if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot2_debug_cut == 8) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 410.0f;
            }
            return;
        }
        warpgroup::sync(6);
        if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot2_debug_cut == 9) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 411.0f;
            }
            return;
        }
        if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot2_debug_cut <= 9) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 400.0f;
            }
            return;
        }
        if (k_block_idx == 1) {
            warpgroup::tma::store_async_read_wait<0>();
            warpgroup::sync(6);
            if (combo_drain_leader) {
                warpgroup::tma::cluster::arrive(combo_dc_p3_outputs_finished[slot], 0, 1);
            }
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 388.0f;
            }
            continue;
        }

        #pragma unroll
        for (int combo_epi = 0; combo_epi < C::EPI_PIPE_DEPTH; ++combo_epi) {
            combo_dc_rt D_reg_fl;
            combo_dc_rt_bf D_reg_bf;
            warpgroup::tma::store_async_read_wait<0>();
            warpgroup::sync(6);
            warpgroup::load_async(
                D_reg_fl,
                combo_dc_tm.template subtile<full_tt_fl<C::Nb / C::EPI_PIPE_DEPTH>>(
                    0,
                    combo_epi * (C::Nb / C::EPI_PIPE_DEPTH)));
            if (combo_epi == 0) {
                COMBO_DC_STAGE_RETURN_DRAIN(14, "drain_after_first_load_async");
            }
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_drain_debug_cut == 1 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 451.0f;
                }
                return;
            }
            if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot3_drain_debug_cut == 1 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 431.0f;
                }
                return;
            }
            if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot2_debug_cut == 10 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 412.0f;
                }
                return;
            }
            tensor_load_wait();
            tensor_before_thread_sync();
            warpgroup::sync(6);
            if (combo_epi == 0) {
                COMBO_DC_STAGE_RETURN_DRAIN(15, "drain_after_post_load_sync");
            }
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_drain_debug_cut == 2 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 452.0f;
                }
                return;
            }
            if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot3_drain_debug_cut == 2 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 432.0f;
                }
                return;
            }
            if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot2_debug_cut == 11 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 413.0f;
                }
                return;
            }
            warp::mul(D_reg_fl, D_reg_fl, combo_dc_scale);
            warp::copy(D_reg_bf, D_reg_fl);
            warpgroup::store(combo_dc_output_stage.dC, D_reg_bf);
            warpgroup::sync(6);
            warpgroup::tma::store_add_async(
                g.dC_out, combo_dc_output_stage.dC,
                {col_block_idx, k_block_idx * C::EPI_PIPE_DEPTH + combo_epi});
            tensor_after_thread_sync();
            if (combo_epi == 0) {
                COMBO_DC_STAGE_RETURN_DRAIN(16, "drain_after_first_writeout");
            }
            if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot4_drain_debug_cut == 3 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 453.0f;
                }
                return;
            }
            if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot3_drain_debug_cut == 3 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 433.0f;
                }
                return;
            }
            if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
                combo_dc_slot2_debug_cut == 12 && combo_epi == 0) {
                if (combo_drain_leader) {
                    g.G_sg_row[{0}] = 414.0f;
                }
                return;
            }
        }
        warpgroup::tma::store_async_read_wait<0>();
        warpgroup::sync(6);
        if (combo_drain_leader) {
            warpgroup::tma::cluster::arrive(combo_dc_p3_outputs_finished[slot], 0, 1);
        }
        COMBO_DC_STAGE_RETURN_DRAIN(17, "drain_after_outputs_finished_arrive");
        if (k_block_idx == 4 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot4_drain_debug_cut == 4) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 454.0f;
            }
            return;
        }
        if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot3_drain_debug_cut == 4) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 434.0f;
            }
            return;
        }
        if (k_block_idx == 2 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot2_debug_cut >= 13) {
            if (combo_drain_leader) {
                combo_dc_debug_mailbox[0] = 1;
                __threadfence_block();
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                g.G_sg_row[{0}] = 415.0f;
            }
            if constexpr (combo_dc_slot2_debug_cut == 13) {
                return;
            }
        }
        if (k_block_idx == 3 && combo_dc_debug_mailbox != nullptr &&
            combo_dc_slot3_drain_debug_cut == 0) {
            if (combo_drain_leader) {
                g.G_sg_row[{0}] = 430.0f;
            }
            return;
        }
    }
    warpgroup::tma::store_async_read_wait<0>();
    warpgroup::sync(6);
    if (combo_dc_debug_mailbox != nullptr && combo_drain_leader) {
        COMBO_DC_STAGE_RETURN_DRAIN(18, "drain_after_final_store_wait");
        if (combo_dc_standalone_trace_verbose && cluster_id == 0 && cta_id == 0) {
            printf("combo_dc tmem drain complete addr=%u standalone=%d try_deprov=%d\n",
                   combo_dc_tmem_addr, combo_dc_debug_mailbox != nullptr ? 1 : 0,
                   combo_dc_standalone_try_deprovision ? 1 : 0);
        }
        if (combo_dc_standalone_try_deprovision || combo_dc_standalone_stage == 19) {
            tm_allocator.deprovision();
            if (combo_dc_standalone_stage == 19 && cluster_id == 0 && cta_id == 0) {
                printf("combo_dc stage 19 drain_after_deprovision addr=%u\n",
                       combo_dc_tmem_addr);
                return;
            }
            if (combo_dc_standalone_trace_verbose && cluster_id == 0 && cta_id == 0) {
                printf("combo_dc standalone deprovisioned addr=%u\n", combo_dc_tmem_addr);
            }
        }
    }
#undef COMBO_DC_STAGE_RETURN_DRAIN
}

template <typename C>
__device__ inline void backward_kernel_v6_streaming_5wg_tail_dconly_from_global(
    const globals_5wg<C>& g)
{
    using G = globals_5wg<C>;

    __shared__ uint32_t combo_dc_tmem_addr;
    __shared__ volatile int combo_dc_debug_mailbox[3];
    __shared__ semaphore combo_dc_debug_sem;
    __shared__ semaphore combo_dc_tmem_provisioned;
    __shared__ semaphore combo_dc_p3_e_tiles_arrived;
    __shared__ semaphore combo_dc_p3_e_scales_arrived;
    __shared__ semaphore combo_dc_p3_inputs_finished;
    __shared__ semaphore combo_dc_p3_outputs_arrived[3];
    __shared__ semaphore combo_dc_p3_outputs_committed[3];
    __shared__ semaphore combo_dc_p3_outputs_finished[3];

    if (threadIdx.x == 0) {
        combo_dc_debug_mailbox[0] = 0;
        combo_dc_debug_mailbox[1] = 0;
        combo_dc_debug_mailbox[2] = 0;
        g.E_col.template prefetch_tma<typename G::combo_p3_E_tile>();
        g.E_col_sc.template prefetch_tma<typename G::combo_p3_E_sc_tile>();
        g.dC_out.template prefetch_tma<typename G::combo_dC_tile>();
        init_semaphore(combo_dc_debug_sem, 0, 1);
        init_semaphore(combo_dc_tmem_provisioned, 0, 1);
        init_semaphore(combo_dc_p3_e_tiles_arrived, 0, 1);
        init_semaphore(combo_dc_p3_e_scales_arrived, 0, 1);
        init_semaphore(combo_dc_p3_inputs_finished, 0, 1);
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            init_semaphore(combo_dc_p3_outputs_arrived[i], 0, 1);
            init_semaphore(combo_dc_p3_outputs_committed[i], 0, 1);
            init_semaphore(combo_dc_p3_outputs_finished[i], 0, C::CLUSTER_SIZE);
        }
    }
    __syncthreads();

    everyone::tma::cluster::arrive_aligned();
    everyone::tma::cluster::wait_aligned();

    const int warpgroup_id = warpgroup::groupid();
    if (warpgroup_id == 2) {
        issue_combo_dc_from_global_bridge<C>(
            g, cluster_ctarank(), clusterIdx().x, combo_dc_debug_mailbox, &combo_dc_debug_sem,
            combo_dc_tmem_addr, combo_dc_tmem_provisioned,
            combo_dc_p3_e_tiles_arrived, combo_dc_p3_e_scales_arrived,
            combo_dc_p3_inputs_finished,
            combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_committed,
            combo_dc_p3_outputs_finished);
    } else if (warpgroup_id == 4) {
        drain_combo_dc_from_global_bridge<C>(
            g, cluster_ctarank(), clusterIdx().x, combo_dc_debug_mailbox,
            combo_dc_tmem_addr, combo_dc_tmem_provisioned,
            combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
    }

    // Probe second-slot issue without forcing consume/teardown of every slot.
}

template <typename C, int STATIC_COMBO_MODE>
__device__ inline void backward_kernel_v6_streaming_5wg_bridge(const globals_5wg<C>& g) {
    using G = globals_5wg<C>;
    using FrontendC = typename C::base;
    using FrontendG = nvfp4_cce_backward_v3::globals_3wg<FrontendC>;
    static_assert(sizeof(FrontendG) == sizeof(globals_5wg<C>),
                  "v6 bridge alias expects the 5-WG and public-v3 globals layouts to match");

    __shared__ volatile int frontend_done[3];
    __shared__ semaphore combo_bridge_de_done;
    __shared__ volatile int combo_bridge_de_done_flag;

    __shared__ uint32_t combo_de_tmem_addr;
    __shared__ semaphore combo_de_tmem_provisioned;
    __shared__ semaphore combo_de_p3_c_tiles_arrived;
    __shared__ semaphore combo_de_p3_c_scales_arrived;
    __shared__ semaphore combo_de_p3_inputs_finished;
    __shared__ semaphore combo_de_p3_outputs_arrived[2];
    __shared__ semaphore combo_de_p3_outputs_finished[2];

    __shared__ uint32_t combo_dc_tmem_addr;
    __shared__ semaphore combo_dc_tmem_provisioned;
    __shared__ semaphore combo_dc_p3_e_tiles_arrived;
    __shared__ semaphore combo_dc_p3_e_scales_arrived;
    __shared__ semaphore combo_dc_p3_inputs_finished;
    __shared__ semaphore combo_dc_p3_outputs_arrived[3];
    __shared__ semaphore combo_dc_p3_outputs_committed[3];
    __shared__ semaphore combo_dc_p3_outputs_finished[3];

    if (threadIdx.x == 0) {
        frontend_done[0] = 0;
        frontend_done[1] = 0;
        frontend_done[2] = 0;
        combo_bridge_de_done_flag = 0;
        if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY ||
                      STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
            g.C_col.template prefetch_tma<typename G::combo_p3_C_tile>();
            g.C_col_sc.template prefetch_tma<typename G::combo_p3_C_sc_tile>();
            g.dE_out.template prefetch_tma<typename G::combo_dE_tile>();
            init_semaphore(combo_de_tmem_provisioned, 0, 1);
            init_semaphore(combo_de_p3_c_tiles_arrived, 0, 1);
            init_semaphore(combo_de_p3_c_scales_arrived, 0, 1);
            init_semaphore(combo_de_p3_inputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                init_semaphore(combo_de_p3_outputs_arrived[i], 0, 1);
                init_semaphore(combo_de_p3_outputs_finished[i], 0, C::CLUSTER_SIZE);
            }
        }
        if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY ||
                      STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
            g.E_col.template prefetch_tma<typename G::combo_p3_E_tile>();
            g.E_col_sc.template prefetch_tma<typename G::combo_p3_E_sc_tile>();
            g.dC_out.template prefetch_tma<typename G::combo_dC_tile>();
            init_semaphore(combo_dc_tmem_provisioned, 0, 1);
            init_semaphore(combo_dc_p3_e_tiles_arrived, 0, 1);
            init_semaphore(combo_dc_p3_e_scales_arrived, 0, 1);
            init_semaphore(combo_dc_p3_inputs_finished, 0, 1);
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                init_semaphore(combo_dc_p3_outputs_arrived[i], 0, 1);
                init_semaphore(combo_dc_p3_outputs_committed[i], 0, 1);
                init_semaphore(combo_dc_p3_outputs_finished[i], 0, C::CLUSTER_SIZE);
            }
        }
        if constexpr (STATIC_COMBO_MODE != G::COMBO_MODE_GONLY) {
            init_semaphore(combo_bridge_de_done, 0, 1);
        }
    }
    __syncthreads();

    const FrontendG& g_front = reinterpret_cast<const FrontendG&>(g);
    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const bool wg_leader = (warpgroup::warpid() == 0) && (warp::laneid() == 0);

    nvfp4_cce_backward_v3::backward_kernel_v3_streaming_3wg_impl<
        FrontendC, true, true, -1, 3>(g_front);

    if (warpgroup_id < 3 && wg_leader) {
        __threadfence_block();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        frontend_done[warpgroup_id] = 1;
    }
    if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY) {
        if (warpgroup_id == 2) {
            if (wg_leader) {
                while (frontend_done[2] != 7) {}
                g.G_sg_row[{0}] = 293.0f;
            }
            return;
        }
        if (warpgroup_id == 0 && wg_leader) {
            __threadfence_block();
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            frontend_done[2] = 7;
        }
        return;
    }
    if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY ||
                  STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
        everyone::tma::cluster::arrive_aligned();
        everyone::tma::cluster::wait_aligned();
    }

    if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_GONLY) {
        if (warpgroup_id < 2) {
            return;
        }
        return;
    } else if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DEONLY) {
        return;
    } else if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_DCONLY) {
        if (warpgroup_id < 2) {
            return;
        }
        if (warpgroup_id == 2) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(4);
            issue_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id, nullptr, nullptr,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_e_tiles_arrived, combo_dc_p3_e_scales_arrived,
                combo_dc_p3_inputs_finished,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_committed,
                combo_dc_p3_outputs_finished);
            wait(combo_bridge_de_done, 0);
        } else if (warpgroup_id == 4) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(6);
            drain_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id, nullptr,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
            if (wg_leader) {
                arrive(combo_bridge_de_done);
            }
        }
        return;
    } else if constexpr (STATIC_COMBO_MODE == G::COMBO_MODE_FULL) {
        if (warpgroup_id < 2) {
            return;
        }
        if (warpgroup_id == 2) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(4);
            issue_combo_de_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_de_tmem_addr, combo_de_tmem_provisioned,
                combo_de_p3_c_tiles_arrived, combo_de_p3_c_scales_arrived,
                combo_de_p3_inputs_finished,
                combo_de_p3_outputs_arrived, combo_de_p3_outputs_finished);
            wait(combo_bridge_de_done, 0);
            issue_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id, nullptr, nullptr,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_e_tiles_arrived, combo_dc_p3_e_scales_arrived,
                combo_dc_p3_inputs_finished,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_committed,
                combo_dc_p3_outputs_finished);
        } else if (warpgroup_id == 3) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(5);
            drain_combo_de_from_global_bridge<C>(
                g, cta_id, cluster_id,
                combo_de_tmem_addr, combo_de_tmem_provisioned,
                combo_de_p3_inputs_finished,
                combo_de_p3_outputs_arrived, combo_de_p3_outputs_finished);
            if (wg_leader) {
                arrive(combo_bridge_de_done);
            }
        } else if (warpgroup_id == 4) {
            while (frontend_done[0] == 0 || frontend_done[1] == 0 || frontend_done[2] == 0) {}
            warpgroup::sync(6);
            wait(combo_bridge_de_done, 0);
            drain_combo_dc_from_global_bridge<C>(
                g, cta_id, cluster_id, nullptr,
                combo_dc_tmem_addr, combo_dc_tmem_provisioned,
                combo_dc_p3_outputs_arrived, combo_dc_p3_outputs_finished);
        }
        return;
    }
}

template <typename C>
__device__ inline void backward_kernel_v6_streaming_5wg_gonly(const globals_5wg<C>& g) {
    backward_kernel_v6_streaming_5wg_bridge<C, globals_5wg<C>::COMBO_MODE_GONLY>(g);
}

template <typename C, int STATIC_COMBO_MODE>
__device__ inline void backward_kernel_v6_streaming_5wg_mode(const globals_5wg<C>& g) {
    static_assert(C::NUM_WARPGROUPS == 5,
                  "v6 dedicated combo entrypoint is reserved for 5-WG configs.");
    backward_kernel_v6_streaming_5wg_bridge<C, STATIC_COMBO_MODE>(g);
}

template <typename C>
__device__ inline void backward_kernel_v6_streaming_5wg(const globals_5wg<C>& g) {
    static_assert(C::NUM_WARPGROUPS == 5,
                  "v6 dedicated combo entrypoint is reserved for 5-WG configs.");
    backward_kernel_v6_streaming_5wg_bridge<C, globals_5wg<C>::COMBO_MODE_FULL>(g);
}

}  // namespace nvfp4_cce_backward_v6
