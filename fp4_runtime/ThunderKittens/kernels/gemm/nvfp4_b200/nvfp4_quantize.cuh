#pragma once
// ================================================================
// NVFP4 Quantization Kernels
// Includes: absmax, quantize, fp8 NaN fixup, fp4<->fp32 utilities.
// ================================================================

#include "kittens.cuh"

using namespace kittens;

namespace nvfp4_quantize {

struct absmax_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 148 * 4;
    static constexpr int NUM_WARPGROUPS = 4;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct quantize_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int TILE_M = 128;      // This should not change
    static constexpr int TILE_N = 128;      // This should not change
    static constexpr int K_BLOCK_SIZE = 16; // This should not change

    using A_bf16_tile  = st_bf<TILE_M, TILE_N, false>;
    using A_fp4x2_tile = st_fp4e2m1_2<TILE_M, TILE_N/2, false>;
    using A_sc_vec     = sv_hf<256>;

    using A_bf16_gl      = gl<bf16,      1,  1, -1, -1, A_bf16_tile>;
    using A_fp4x2_gl     = gl<fp4e2m1_2, 1,  1, -1, -1, A_fp4x2_tile>;
    using A_sc_gl        = gl<half,      1, -1, -1, 256, A_sc_vec>;
    using A_sc_global_gl = gl<float,     1,  1,  1,  1>;

    A_bf16_gl      A_bf16;      // M x N
    A_fp4x2_gl     A_fp4x2;     // M x (N // 2)
    A_sc_gl        A_sc;        // (M // 128) x (N // 64) x 512
    A_sc_global_gl A_sc_global; // (1,)

    __host__ inline dim3 grid() const {
        return dim3(A_bf16.cols() / TILE_N, A_bf16.rows() / TILE_M);
    }
    __host__ inline int dynamic_shared_memory() const {
        return TILE_M * TILE_N * sizeof(bf16) + 1024;
    }
};

__global__ void zero_kernel(const globals g) {
    g.A_sc_global.raw_ptr[0] = 0.0f;
}

__global__ void absmax_kernel(const globals g) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;
    const size_t numel = g.A_bf16.rows() * g.A_bf16.cols();

    bf16 local_max = __float2bfloat16(0.0f);
    bf16_2 *base_ptr = reinterpret_cast<bf16_2*>(g.A_bf16.raw_ptr);

    for (size_t i = tid; i < numel / 8; i += num_threads) {
        bf16_2 v0, v1, v2, v3;
        asm volatile(
            "ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(*(uint32_t*)&v0), "=r"(*(uint32_t*)&v1), "=r"(*(uint32_t*)&v2), "=r"(*(uint32_t*)&v3)
            : "l"(base_ptr + i*4)
        );

        bf16_2 abs0 = __habs2(v0);
        bf16_2 abs1 = __habs2(v1);
        bf16_2 abs2 = __habs2(v2);
        bf16_2 abs3 = __habs2(v3);

        bf16_2 max01 = __hmax2(abs0, abs1);
        bf16_2 max23 = __hmax2(abs2, abs3);
        bf16_2 max0123 = __hmax2(max01, max23);

        bf16 curr_max = __hmax(max0123.x, max0123.y);
        local_max = __hmax(local_max, curr_max);
    }

    for (size_t i = (numel / 8) * 8 + tid; i < numel; i += num_threads)
        local_max = __hmax(local_max, __habs(g.A_bf16.raw_ptr[i]));

    #pragma unroll
    for (int offset = WARP_THREADS / 2; offset > 0; offset /= 2) {
        uint32_t local_bits = *reinterpret_cast<unsigned short*>(&local_max);
        uint32_t other_bits = __shfl_xor_sync(0xffffffff, local_bits, offset);
        local_max = __hmax(local_max, *reinterpret_cast<bf16*>(&other_bits));
    }

    __shared__ bf16 shared_max[absmax_config::NUM_WARPS];
    if (laneid() == 0) shared_max[warpid()] = local_max;
    __syncthreads();

    if (warpid() == 0) {
        bf16 val = (laneid() < absmax_config::NUM_WARPS) ? shared_max[laneid()] : __float2bfloat16(0.0f);

        #pragma unroll
        for (int offset = absmax_config::NUM_WARPS / 2; offset > 0; offset /= 2) {
            uint32_t val_bits = *reinterpret_cast<unsigned short*>(&val);
            uint32_t other_bits = __shfl_xor_sync(0xffffffff, val_bits, offset);
            val = __hmax(val, *reinterpret_cast<bf16*>(&other_bits));
        }

        if (laneid() == 0) {
            float val_fl = __bfloat162float(val); // Positive float values keep bit ordering
            atomicMax(reinterpret_cast<uint32_t*>(g.A_sc_global.raw_ptr), *reinterpret_cast<uint32_t*>(&val_fl));
        }
    }
}

__global__ void divide_kernel(const globals g) {
    g.A_sc_global.raw_ptr[0] /= 6.0f * 448.0f;
}

template<bool SCALE_2D = false>
__device__ inline void quantize_kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    globals::A_bf16_tile &A_bf16_smem = sm_allocator.allocate<globals::A_bf16_tile>();
    globals::A_fp4x2_tile &A_fp4x2_smem = *reinterpret_cast<globals::A_fp4x2_tile *>(&A_bf16_smem);
    globals::A_sc_vec (&A_sc_smem)[2] = *reinterpret_cast<globals::A_sc_vec(*)[2]>(
        reinterpret_cast<uint64_t>(&A_fp4x2_smem) + sizeof(A_fp4x2_smem));

    // Calculate indices
    const int tid = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect(inputs_arrived, A_bf16_smem);
        tma::load_async(A_bf16_smem, G.A_bf16, {row, col}, inputs_arrived);
    }

    // Fetch pre-calculated global scales
    float s_global_dec = G.A_sc_global[{0}];
    float s_global_enc = 1.0f / fmaxf(s_global_dec, 0.000000000001f);

    // We have 128 threads per block. Each thread handles 1 row of 128 elements.
    const int tile_row = tid;
    constexpr int NUM_K_BLOCKS_HALF = globals::TILE_N / globals::K_BLOCK_SIZE / 2;  // 4
    constexpr int N_PER_K_BLOCK = globals::K_BLOCK_SIZE / 2;                        // 8
    bf16_2 A_bf16_reg[2][NUM_K_BLOCKS_HALF][N_PER_K_BLOCK]; // [col_half][k_block][elem]
    fp8e4m3 A_sc_reg[2][NUM_K_BLOCKS_HALF];                 // [col_half][k_block]

    // Wait for the inputs to arrive
    __syncthreads();
    wait(inputs_arrived, 0);

    // Load input matrix from shared memory (custom swizzling to avoid bank conflicts)
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + tid/8)%NUM_K_BLOCKS_HALF + col_half*NUM_K_BLOCKS_HALF;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int tile_col = k_block_idx*globals::K_BLOCK_SIZE + ((tid+j)*2)%globals::K_BLOCK_SIZE;
                const int offset = (tile_row*globals::TILE_N + tile_col) * sizeof(bf16);
                move<bf16_2>::lds(A_bf16_reg[col_half][i][j], static_cast<uint32_t>(__cvta_generic_to_shared(&A_bf16_smem)) + offset);
            }
        }
    }
    __syncthreads();

    // Perform NVFP4 quantization
    #pragma unroll
    for (int col_half = 0; col_half < 2; col_half++) {
        // Calculate absolute maximum for each K block
        float amax[NUM_K_BLOCKS_HALF];
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + tid/8) % NUM_K_BLOCKS_HALF;
            bf16_2 _amax = __habs2(A_bf16_reg[col_half][i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_K_BLOCK; j++)
                _amax = __hmax2(_amax, __habs2(A_bf16_reg[col_half][i][j]));
            amax[k_block_idx] = __bfloat162float(__hmax(_amax.x, _amax.y));
        }

        // For 2D scaling, reduce amax across 16 rows
        if constexpr (SCALE_2D) {
            #pragma unroll
            for (int mask = 8; mask >= 1; mask >>= 1) {
                #pragma unroll
                for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
                    amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffff, amax[i], mask));
            }
        }

        // Compute the local scales
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++)
            A_sc_reg[col_half][i] = __nv_fp8_e4m3(amax[i] / 6.0f * s_global_enc);

        // Quantize input matrix to FP4 and store to shared memory
        #pragma unroll
        for (int i = 0; i < NUM_K_BLOCKS_HALF; i++) {
            const int k_block_idx = (i + tid/8) % NUM_K_BLOCKS_HALF;
            const float s_local_dec = static_cast<float>(A_sc_reg[col_half][k_block_idx]); // choked
            const float s_enc = 1.0f / fmaxf(s_local_dec*s_global_dec, 0.000000000001f);
            const int offset_base = tile_row*globals::TILE_N/2 + (k_block_idx + col_half*NUM_K_BLOCKS_HALF)*globals::K_BLOCK_SIZE/2;
            #pragma unroll
            for (int j = 0; j < N_PER_K_BLOCK; j++) {
                const int offset = offset_base + ((tid+j)&7);
                const float2 scaled = {
                    __bfloat162float(A_bf16_reg[col_half][i][j].x)*s_enc,
                    __bfloat162float(A_bf16_reg[col_half][i][j].y)*s_enc
                };
                asm volatile("{st.shared.b8 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_fp4x2_smem)) + offset)
                       "r"(static_cast<uint32_t>(__nv_cvt_float2_to_fp4x2(scaled, __NV_E2M1, cudaRoundNearest))));
            }
        }
    }

    // Store the scales to shared memory following NVIDIA's scale swizzle layout
    const int scale_offset = (tile_row%32) * 16 + (tile_row/32) * 4;
    asm volatile("{st.shared.b32 [%0], %1;}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem[0])) + scale_offset)
           "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[0][0])));
    asm volatile("{st.shared.b32 [%0], %1;}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&A_sc_smem[1])) + scale_offset)
           "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[1][0])));

    // Store to global memory
    __syncthreads();
    if (tid == 0) {
        tma::store_async(G.A_fp4x2, A_fp4x2_smem, {row, col});
        tma::store_async(G.A_sc, A_sc_smem[0], {row, col*2+0, 0});
        tma::store_async(G.A_sc, A_sc_smem[1], {row, col*2+1, 0});
    }
}

} // namespace nvfp4_quantize

// Standalone kernel: fixup FP8 E4M3 NaN (0x7F/0xFF) → max (0x7E/0xFE = ±448.0)
// Processes 4 bytes (uint32) per thread for efficiency.
__global__ void fp8_nan_fixup_kernel(uint8_t* __restrict__ data, int64_t numel) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // Process 4 bytes at a time
    int64_t idx4 = idx * 4;
    if (idx4 + 3 < numel) {
        uint32_t packed = *reinterpret_cast<uint32_t*>(data + idx4);
        // Check if any byte is NaN (0x7F or 0xFF) and fixup
        bool needs_fix = false;
        uint32_t b0 = (packed >>  0) & 0xFF; if ((b0 & 0x7F) == 0x7F) { b0 = (b0 & 0x80) | 0x7E; needs_fix = true; }
        uint32_t b1 = (packed >>  8) & 0xFF; if ((b1 & 0x7F) == 0x7F) { b1 = (b1 & 0x80) | 0x7E; needs_fix = true; }
        uint32_t b2 = (packed >> 16) & 0xFF; if ((b2 & 0x7F) == 0x7F) { b2 = (b2 & 0x80) | 0x7E; needs_fix = true; }
        uint32_t b3 = (packed >> 24) & 0xFF; if ((b3 & 0x7F) == 0x7F) { b3 = (b3 & 0x80) | 0x7E; needs_fix = true; }
        if (needs_fix) {
            *reinterpret_cast<uint32_t*>(data + idx4) = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    } else {
        // Handle tail elements
        for (int64_t i = idx4; i < numel; i++) {
            if ((data[i] & 0x7F) == 0x7F) data[i] = (data[i] & 0x80) | 0x7E;
        }
    }
}

namespace nvfp4_utils {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1024; // arbitrary
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    using A_fp32_gl = gl<float, 1, 1, -1, -1>;
    using A_fp4x2_gl = gl<fp4e2m1_2, 1, 1, -1, -1>;

    A_fp32_gl A_fp32;
    A_fp4x2_gl A_fp4x2;
};

__device__ inline void fp32_to_fp4x2_kernel(const globals &G) {
    // This kernel is for testing purposes only
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < G.A_fp32.numel() / 2; i += blockDim.x * gridDim.x) {
        float2 A_fp32x2 = {G.A_fp32.raw_ptr[i * 2 + 0], G.A_fp32.raw_ptr[i * 2 + 1]};
        G.A_fp4x2.raw_ptr[i].__x = __nv_cvt_float2_to_fp4x2(A_fp32x2, __NV_E2M1, cudaRoundNearest);
    }
}

__device__ inline void fp4x2_to_fp32_kernel(const globals &G) {
    // This kernel is for testing purposes only
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < G.A_fp32.numel() / 2; i += blockDim.x * gridDim.x) {
        float2 A_fp32x2 = static_cast<float2>(G.A_fp4x2.raw_ptr[i]);
        G.A_fp32.raw_ptr[i * 2 + 0] = A_fp32x2.x;
        G.A_fp32.raw_ptr[i * 2 + 1] = A_fp32x2.y;
    }
}

} // namespace nvfp4_utils
