/*************************************************************************
 * Mixed MXFP4/localCTA carriers.
 *
 * These kernels deliberately expose only the two heterogeneous contracts
 * used by the direction-split experiment:
 *
 *   gradient: localCTA row-SR + MXFP4 column-RHT (plain H, block 16)
 *   weight:   MXFP4 shared-2D row + localCTA shared-2D column
 *
 * Each kernel stages the BF16 source tile once.  There is no second BF16
 * carrier and no hidden RNG reservation.  The gradient kernel consumes the
 * explicit seed/subsequence coordinate supplied by the ranked producer.
 *************************************************************************/

#ifndef TK_MIXED_MXFP4_LOCALCTA_CUH_
#define TK_MIXED_MXFP4_LOCALCTA_CUH_

namespace tk_mixed_mx_localcta {

using MXMode = mxfp4_v3::QuantMode;
using MXInput = mxfp4_v3::InputBuf3D;
using MXOutput = mxfp4_v3::OutputBuf3D;
using LocalInput = tk_localcta::IType3D;
using LocalRowOutput = tk_localcta::OType2x3D;
using LocalColOutput = tk_localcta::OType2xt3D;
using LocalRowScales = tk_localcta::ScalesType2D;
using LocalColScales = tk_localcta::ScalesTypeTr2D;
using transformer_engine::bf16;
using transformer_engine::fp4e2m1x2;
using transformer_engine::dispatch::nvfp4::nvfp4_scale_t;

static constexpr int kThreads = 128;
static constexpr int kInputBytes = 4 * 64 * 64 * static_cast<int>(sizeof(bf16));
static constexpr int kPackedTileBytes = 64 * 32;
static constexpr int kPackedOutputBytes = 2 * kPackedTileBytes;
static constexpr int kLocalScaleBytes = 128 * 8 * static_cast<int>(sizeof(nvfp4_scale_t));
static constexpr int kMxScaleBytes = 128 * 4;
static constexpr int kMixedWeightPackedBytes =
    2 * kPackedOutputBytes + kPackedTileBytes;

// Input + one localCTA row ring + one MX output ring + both scale workspaces.
static constexpr int mixed_grad_shmem_size() {
    return kInputBytes + 2 * kPackedOutputBytes + kLocalScaleBytes +
        kMxScaleBytes + TMA_SHMEM_ALIGNMENT;
}

// Input + localCTA row/column rings + one MX row tile + scale workspaces.
// The mixed ABI never emits an MX column weight, so it does not reserve the
// second MX output tile or an MX column-scale scratch buffer.
static constexpr int mixed_weight_shmem_size() {
    return kInputBytes + kMixedWeightPackedBytes + 2 * kLocalScaleBytes +
        kMxScaleBytes + TMA_SHMEM_ALIGNMENT;
}

__device__ __forceinline__ float reduce_localcta_chunk_amax(
    const bf16* s_in_ptr
) {
    float cta_max = 0.0f;
    #pragma unroll
    for (int t = 0; t < tk_localcta::NUM_TILES; ++t) {
        cta_max = fmaxf(cta_max, tk_localcta::scan_tile_amax(s_in_ptr, t));
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        cta_max = fmaxf(
            cta_max,
            __shfl_xor_sync(0xffffffffu, cta_max, mask));
    }

    __shared__ float warp_max[kThreads / 32];
    __shared__ float chunk_amax;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (lane == 0) {
        warp_max[warp] = cta_max;
    }
    __syncthreads();

    if (warp == 0) {
        cta_max = lane < kThreads / 32 ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int mask = (kThreads / 32) / 2; mask > 0; mask >>= 1) {
            cta_max = fmaxf(
                cta_max,
                __shfl_xor_sync(0xffffffffu, cta_max, mask));
        }
        if (lane == 0) {
            chunk_amax = cta_max;
        }
    }
    __syncthreads();
    return chunk_amax;
}

__device__ __forceinline__ void load_mixed_chunk(
    const CUtensorMap& tensor_map_input,
    MXInput& s_in,
    uint64_t (&in_mbar)[mxfp4_v3::NUM_TILES],
    int block_offset_y,
    int block_offset_x
) {
    const bool leading = threadIdx.x == 0;
    if (leading) {
        #pragma unroll
        for (int t = 0; t < mxfp4_v3::NUM_TILES; ++t) {
            transformer_engine::ptx::mbarrier_init(&in_mbar[t], 1);
        }
        transformer_engine::ptx::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if (leading) {
        #pragma unroll
        for (int t = 0; t < mxfp4_v3::NUM_TILES; ++t) {
            const int tile_y = t / mxfp4_v3::TILES_X;
            const int tile_x = t % mxfp4_v3::TILES_X;
            transformer_engine::ptx::mbarrier_arrive_expect_tx(
                &in_mbar[t],
                mxfp4_v3::TILE_DIM * mxfp4_v3::TILE_DIM * sizeof(bf16));
            transformer_engine::ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t*>(&s_in[t]),
                reinterpret_cast<const uint64_t*>(&tensor_map_input),
                block_offset_x + tile_x * mxfp4_v3::TILE_DIM,
                block_offset_y + tile_y * mxfp4_v3::TILE_DIM,
                &in_mbar[t]);
        }
    }

    #pragma unroll
    for (int t = 0; t < mxfp4_v3::NUM_TILES; ++t) {
        transformer_engine::ptx::mbarrier_wait_parity_acquire_cta_shared_cta(
            &in_mbar[t], 0);
    }
    __syncthreads();
}

__device__ __forceinline__ void invalidate_mixed_mbars(
    uint64_t (&in_mbar)[mxfp4_v3::NUM_TILES]
) {
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int t = 0; t < mxfp4_v3::NUM_TILES; ++t) {
            transformer_engine::ptx::mbarrier_invalid(&in_mbar[t]);
        }
    }
}

// The launch bounds are intentional: the mixed producer must retain at least
// two resident CTAs while the post-build resource contract separately rejects
// any stack/local frame.
__global__ void __launch_bounds__(kThreads, 2)
mixed_grad_localcta_row_mx_col_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input0,
    const __grid_constant__ CUtensorMap tensor_map_input1,
    const __grid_constant__ CUtensorMap tensor_map_local_row_output,
    const __grid_constant__ CUtensorMap tensor_map_local_row_scales,
    const __grid_constant__ CUtensorMap tensor_map_mx_col_output,
    uint8_t* __restrict__ mx_col_scales_out,
    float* __restrict__ local_row_sg_chunks,
    const int64_t rows,
    const int64_t cols,
    const int split_blocks_x,
    const uint64_t rng_seed,
    const uint64_t rng_subsequence_base
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = threadIdx.x == 0;
    const int cta_x = static_cast<int>(blockIdx.x);
    const int cta_y = static_cast<int>(blockIdx.y);
    const int blocks_x = static_cast<int>(cols / 128);
    const int block_offset_y = cta_y * 128;
    const int block_offset_x = cta_x * 128;
    const bool split2 = split_blocks_x > 0;
    const bool use_second_input = split2 && cta_x >= split_blocks_x;
    const int input_cta_x = use_second_input ? cta_x - split_blocks_x : cta_x;
    const int input_block_offset_x = input_cta_x * 128;
    const CUtensorMap* tensor_map_input =
        use_second_input ? &tensor_map_input1 : &tensor_map_input0;
    const uint64_t chunk_id =
        static_cast<uint64_t>(cta_y) * blocks_x + cta_x;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem =
        transformer_engine::common::align_smem_ptr_per_TMA_requirements(
            dynamic_shmem);
    bf16* s_in_ptr = reinterpret_cast<bf16*>(dshmem);
    fp4e2m1x2* local_row_out_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + kInputBytes);
    fp4e2m1x2* mx_col_out_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + kInputBytes + kPackedOutputBytes);
    nvfp4_scale_t* local_row_sc_ptr = reinterpret_cast<nvfp4_scale_t*>(
        dshmem + kInputBytes + 2 * kPackedOutputBytes);
    uint8_t* mx_col_sc_ptr = reinterpret_cast<uint8_t*>(
        dshmem + kInputBytes + 2 * kPackedOutputBytes + kLocalScaleBytes);

    auto& s_in = *reinterpret_cast<MXInput*>(s_in_ptr);
    auto& local_row_out = *reinterpret_cast<LocalRowOutput*>(local_row_out_ptr);
    auto& mx_col_out = *reinterpret_cast<MXOutput*>(mx_col_out_ptr);
    (void)local_row_out;
    (void)mx_col_out;

    __shared__ uint64_t in_mbar[mxfp4_v3::NUM_TILES];
    load_mixed_chunk(
        *tensor_map_input, s_in, in_mbar,
        block_offset_y, input_block_offset_x);

    const float chunk_amax = reduce_localcta_chunk_amax(s_in_ptr);
    const float local_s_enc =
        tk_localcta::compute_localcta_encode_scaling_factor_FP4(chunk_amax);
    const float local_sg =
        chunk_amax / tk_localcta::localcta_global_scale_num();
    if (leading) {
        local_row_sg_chunks[chunk_id] = local_sg;
    }

    #pragma unroll
    for (int t = 0; t < mxfp4_v3::NUM_TILES; ++t) {
        const int stage_y = t / mxfp4_v3::TILES_X;
        const int stage_x = t % mxfp4_v3::TILES_X;
        const int local_buffer = t & 1;
        const int mx_buffer = t & 1;

        // Both output carriers have two tile buffers.  Keep at most one
        // prior tile's store group in flight before reusing a buffer, as in
        // the proven direct-localCTA producer lifetime pattern.
        if (t > 0 && leading) {
            transformer_engine::ptx::cp_async_bulk_wait_group_read<1>();
        }
        __syncthreads();

        const uint64_t local_subsequence =
            rng_subsequence_base +
            ((chunk_id * tk_localcta::NUM_TILES + t) * 2ull) *
                tk_localcta::THREADS +
            threadIdx.x;
        tk_localcta::LocalCTARNGState local_rng;
        local_rng.init(rng_seed, local_subsequence, 0);
        uint4 local_random = make_uint4(0, 0, 0, 0);
        int local_random_index = 4;
        tk_localcta::rowwise_scaling_opt<
            true, false, true, false, false, false, false, false, false>(
                s_in_ptr,
                local_row_out_ptr,
                local_row_sc_ptr,
                local_s_enc,
                stage_y,
                stage_x,
                t,
                local_buffer,
                local_rng,
                local_random,
                local_random_index,
                0,
                block_offset_y,
                block_offset_x);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            transformer_engine::ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_local_row_output),
                block_offset_x + stage_x * 64,
                block_offset_y + stage_y * 64,
                reinterpret_cast<uint64_t*>(&local_row_out[local_buffer]));
        }
        __syncthreads();

        // Match the completed MXFP4+RHT Wgrad carrier exactly: encode-centric
        // E8M0, normalized H32 and the compile-time fixed 0x2817 sign motif.
        // The BF16 tile is already resident for the paired localCTA row
        // producer, so this adds no extra load or launch.  Data/scale SR stay
        // disabled on the MX column carrier.
        mxfp4_v3::mx_colwise_quantize_direct_opt<
            MXMode::ENCODE, false, false, true, 32, true, false>(
                s_in_ptr,
                mx_col_out_ptr,
                mx_col_sc_ptr,
                stage_x,
                stage_y,
                t,
                mx_buffer,
                0,
                0,
                block_offset_y,
                block_offset_x,
                0);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            transformer_engine::ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_mx_col_output),
                block_offset_y + stage_y * 64,
                block_offset_x + stage_x * 64,
                reinterpret_cast<uint64_t*>(&mx_col_out[mx_buffer]));
            transformer_engine::ptx::cp_async_bulk_commit_group();
        }
        __syncthreads();
    }

    if (leading) {
        transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();

    tk_localcta::swizzle_scales_row_inplace(
        local_row_sc_ptr, tk_localcta::SCALES_PER_CHUNK_X);
    transformer_engine::ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    if (leading) {
        tk_localcta::tma_store_scales_2x512(
            tensor_map_local_row_scales,
            local_row_sc_ptr,
            cta_y,
            cta_x * 2 * 256);
    }
    mxfp4_v3::write_scales_swizzled(
        mx_col_sc_ptr,
        mx_col_scales_out,
        cta_y,
        cta_x,
        static_cast<int>(rows / 128));
    if (leading) {
        transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
    invalidate_mixed_mbars(in_mbar);
#else
    NVTE_DEVICE_ERROR("mixed MXFP4/localCTA carrier requires SM100 or newer");
#endif
}

__global__ void __launch_bounds__(kThreads, 2)
mixed_weight_mx_row_localcta_col_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_mx_row_output,
    const __grid_constant__ CUtensorMap tensor_map_local_col_output,
    const __grid_constant__ CUtensorMap tensor_map_local_col_scales,
    uint8_t* __restrict__ mx_row_scales_out,
    float* __restrict__ local_col_sg_chunks,
    const int64_t rows,
    const int64_t cols
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    const bool leading = threadIdx.x == 0;
    const int cta_x = static_cast<int>(blockIdx.x);
    const int cta_y = static_cast<int>(blockIdx.y);
    const int blocks_y = static_cast<int>(rows / 128);
    const int block_offset_y = cta_y * 128;
    const int block_offset_x = cta_x * 128;

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem =
        transformer_engine::common::align_smem_ptr_per_TMA_requirements(
            dynamic_shmem);
    bf16* s_in_ptr = reinterpret_cast<bf16*>(dshmem);
    fp4e2m1x2* local_row_out_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + kInputBytes);
    fp4e2m1x2* local_col_out_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + kInputBytes + kPackedOutputBytes);
    fp4e2m1x2* mx_out_ptr = reinterpret_cast<fp4e2m1x2*>(
        dshmem + kInputBytes + 2 * kPackedOutputBytes);
    nvfp4_scale_t* local_row_sc_ptr = reinterpret_cast<nvfp4_scale_t*>(
        dshmem + kInputBytes + kMixedWeightPackedBytes);
    nvfp4_scale_t* local_col_sc_ptr = reinterpret_cast<nvfp4_scale_t*>(
        dshmem + kInputBytes + kMixedWeightPackedBytes + kLocalScaleBytes);
    uint8_t* mx_row_sc_ptr = reinterpret_cast<uint8_t*>(
        dshmem + kInputBytes + kMixedWeightPackedBytes + 2 * kLocalScaleBytes);

    auto& s_in = *reinterpret_cast<MXInput*>(s_in_ptr);
    auto& local_col_out = *reinterpret_cast<LocalColOutput*>(local_col_out_ptr);
    auto& mx_out = *reinterpret_cast<MXOutput*>(mx_out_ptr);

    __shared__ uint64_t in_mbar[mxfp4_v3::NUM_TILES];
    load_mixed_chunk(
        tensor_map_input, s_in, in_mbar, block_offset_y, block_offset_x);

    const float chunk_amax = reduce_localcta_chunk_amax(s_in_ptr);
    const float local_s_enc =
        tk_localcta::compute_localcta_encode_scaling_factor_FP4(chunk_amax);
    const float local_sg =
        chunk_amax / tk_localcta::localcta_global_scale_num();
    if (leading) {
        local_col_sg_chunks[cta_x * blocks_y + cta_y] = local_sg;
    }

    #pragma unroll
    for (int t = 0; t < mxfp4_v3::NUM_TILES; ++t) {
        const int stage_y = t / mxfp4_v3::TILES_X;
        const int stage_x = t % mxfp4_v3::TILES_X;
        const int local_buffer = t & 1;

        tk_localcta::weight_2d_scaling<true>(
            s_in_ptr,
            local_row_out_ptr,
            local_col_out_ptr,
            local_row_sc_ptr,
            local_col_sc_ptr,
            local_s_enc,
            stage_y,
            stage_x,
            t,
            local_buffer,
            local_buffer);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            transformer_engine::ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_local_col_output),
                block_offset_y + stage_y * 64,
                block_offset_x + stage_x * 64,
                reinterpret_cast<uint64_t*>(&local_col_out[local_buffer]));
            transformer_engine::ptx::cp_async_bulk_commit_group();
            transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();

        mxfp4_v3::mx_weight_2d_quantize<false>(
            s_in_ptr,
            mx_out_ptr,
            mx_row_sc_ptr,
            nullptr,
            stage_y,
            stage_x,
            t,
            0,
            0);
        transformer_engine::ptx::fence_proxy_async_shared_cta();
        __syncthreads();
        if (leading) {
            transformer_engine::ptx::cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_mx_row_output),
                block_offset_x + stage_x * 64,
                block_offset_y + stage_y * 64,
                reinterpret_cast<uint64_t*>(&mx_out[0]));
            transformer_engine::ptx::cp_async_bulk_commit_group();
            transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }

    tk_localcta::swizzle_scales_col_inplace(
        local_col_sc_ptr, tk_localcta::SCALES_PER_CHUNK_Y);
    transformer_engine::ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    if (leading) {
        tk_localcta::tma_store_scales_2x512(
            tensor_map_local_col_scales,
            local_col_sc_ptr,
            cta_x,
            cta_y * 2 * 256);
    }
    mxfp4_v3::write_scales_swizzled(
        mx_row_sc_ptr,
        mx_row_scales_out,
        cta_x,
        cta_y,
        static_cast<int>(cols / 128));
    if (leading) {
        transformer_engine::ptx::cp_async_bulk_wait_group_read<0>();
    }
    __syncthreads();
    invalidate_mixed_mbars(in_mbar);
#else
    NVTE_DEVICE_ERROR("mixed MXFP4/localCTA weight carrier requires SM100 or newer");
#endif
}

}  // namespace tk_mixed_mx_localcta

#endif  // TK_MIXED_MXFP4_LOCALCTA_CUH_
