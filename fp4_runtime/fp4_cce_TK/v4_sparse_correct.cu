#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

bool use_sparse_vec8() {
    const char* mode = std::getenv("FP4_CCE_V4_SPARSE_VEC8");
    return mode == nullptr || std::strcmp(mode, "0") != 0;
}

int sparse_threads(const char* name, int default_threads) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_threads;
    }
    char* end = nullptr;
    const long threads = std::strtol(raw, &end, 10);
    TORCH_CHECK(
        end != raw && *end == '\0' &&
            (threads == 128 || threads == 256 || threads == 512),
        name,
        " must be 128, 256, or 512");
    return static_cast<int>(threads);
}

struct alignas(16) BFloat16x8 {
    __nv_bfloat162 values[4];
};

__device__ __forceinline__ void unpack_bfloat16x8(
    const BFloat16x8& packed,
    float (&values)[8]
) {
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 converted = __bfloat1622float2(packed.values[pair]);
        values[pair * 2] = converted.x;
        values[pair * 2 + 1] = converted.y;
    }
}

__device__ __forceinline__ BFloat16x8 pack_bfloat16x8(
    const float (&values)[8]
) {
    BFloat16x8 packed;
    #pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        packed.values[pair] = __floats2bfloat162_rn(
            values[pair * 2], values[pair * 2 + 1]);
    }
    return packed;
}

template <bool TARGET_SPLIT, int TOPK_SPLIT>
__global__ void sparse_correct_kernel(
    __nv_bfloat16* __restrict__ dE,
    __nv_bfloat16* __restrict__ dC,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const float* __restrict__ target_probs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_probs,
    const float* __restrict__ scale_ptr,
    int64_t M,
    int64_t K,
    int64_t ignore_index
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = M * K;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / K;
    const int64_t col = idx - row * K;
    const int64_t target = targets[row];
    if (target == ignore_index) {
        return;
    }

    const float scale = *scale_ptr;
    const float correction =
        TARGET_SPLIT ? (1.0f - target_probs[row]) : 1.0f;
    const float w_val = __bfloat162float(weight[target * K + col]);
    float corrected_dE =
        __bfloat162float(dE[idx]) - w_val * scale * correction;
    if constexpr (TOPK_SPLIT > 0) {
        #pragma unroll
        for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
            const int64_t split_offset =
                static_cast<int64_t>(row) * TOPK_SPLIT + rank;
            const int topk = topk_indices[split_offset];
            if (topk >= 0) {
                const float topk_correction = topk_probs[split_offset];
                const float topk_w_val = __bfloat162float(
                    weight[static_cast<int64_t>(topk) * K + col]);
                corrected_dE += topk_w_val * scale * topk_correction;
            }
        }
    }

    dE[idx] = __float2bfloat16(corrected_dE);
}

__global__ void sparse_correct_scaled_dE_kernel(
    __nv_bfloat16* __restrict__ dE,
    __nv_bfloat16* __restrict__ dC,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const float* __restrict__ scale_ptr,
    int64_t M,
    int64_t K,
    int64_t ignore_index
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = M * K;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / K;
    const int64_t col = idx - row * K;
    const int64_t target = targets[row];
    const float scale = *scale_ptr;

    float dE_val = __bfloat162float(dE[idx]) * scale;
    if (target != ignore_index) {
        const float w_val = __bfloat162float(weight[target * K + col]);
        dE_val -= w_val * scale;
    }
    dE[idx] = __float2bfloat16(dE_val);
}

template <bool TARGET_SPLIT, int TOPK_SPLIT>
__global__ void sparse_correct_vec2_kernel(
    __nv_bfloat162* __restrict__ dE,
    __nv_bfloat162* __restrict__ dC,
    const __nv_bfloat162* __restrict__ x,
    const __nv_bfloat162* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const float* __restrict__ target_probs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_probs,
    const float* __restrict__ scale_ptr,
    int64_t M,
    int64_t K2,
    int64_t ignore_index
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = M * K2;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / K2;
    const int64_t pair = idx - row * K2;
    const int64_t target = targets[row];
    if (target == ignore_index) {
        return;
    }

    const float scale = *scale_ptr;
    const float correction =
        TARGET_SPLIT ? (1.0f - target_probs[row]) : 1.0f;
    const float2 w_val = __bfloat1622float2(weight[target * K2 + pair]);
    const float2 dE_val = __bfloat1622float2(dE[idx]);
    float2 corrected_dE = {
        dE_val.x - w_val.x * scale * correction,
        dE_val.y - w_val.y * scale * correction,
    };
    if constexpr (TOPK_SPLIT > 0) {
        #pragma unroll
        for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
            const int64_t split_offset =
                static_cast<int64_t>(row) * TOPK_SPLIT + rank;
            const int topk = topk_indices[split_offset];
            if (topk >= 0) {
                const float topk_correction = topk_probs[split_offset];
                const float2 topk_w_val = __bfloat1622float2(
                    weight[static_cast<int64_t>(topk) * K2 + pair]);
                corrected_dE.x += topk_w_val.x * scale * topk_correction;
                corrected_dE.y += topk_w_val.y * scale * topk_correction;
            }
        }
    }
    dE[idx] = __floats2bfloat162_rn(
        corrected_dE.x,
        corrected_dE.y);
}

template <int TOPK_SPLIT>
__global__ void sparse_correct_target_topk_dE_vec8_kernel(
    BFloat16x8* __restrict__ dE,
    const BFloat16x8* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const float* __restrict__ target_probs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_probs,
    const float* __restrict__ scale_ptr,
    int64_t M,
    int64_t K8,
    int64_t ignore_index
) {
    const int64_t row = blockIdx.x;
    if (row >= M) return;

    __shared__ int selected_rows[TOPK_SPLIT + 1];
    __shared__ float selected_probs[TOPK_SPLIT];
    __shared__ float target_correction;
    __shared__ float scale;
    __shared__ bool valid;

    if (threadIdx.x == 0) {
        const int64_t target = targets[row];
        valid = target != ignore_index;
        selected_rows[0] = valid ? static_cast<int>(target) : -1;
        target_correction = valid ? 1.0f - target_probs[row] : 0.0f;
        scale = *scale_ptr;
    }
    if (threadIdx.x < TOPK_SPLIT) {
        const int64_t offset = row * TOPK_SPLIT + threadIdx.x;
        selected_rows[threadIdx.x + 1] = topk_indices[offset];
        selected_probs[threadIdx.x] = topk_probs[offset];
    }
    __syncthreads();
    if (!valid) return;

    for (int64_t vector = threadIdx.x; vector < K8;
         vector += blockDim.x) {
        const int64_t output_offset = row * K8 + vector;
        float corrected[8];
        unpack_bfloat16x8(dE[output_offset], corrected);

        float selected[8];
        unpack_bfloat16x8(
            weight[static_cast<int64_t>(selected_rows[0]) * K8 + vector],
            selected);
        #pragma unroll
        for (int element = 0; element < 8; ++element) {
            corrected[element] -=
                selected[element] * scale * target_correction;
        }

        #pragma unroll
        for (int rank = 0; rank < TOPK_SPLIT; ++rank) {
            const int selected_row = selected_rows[rank + 1];
            if (selected_row >= 0) {
                unpack_bfloat16x8(
                    weight[static_cast<int64_t>(selected_row) * K8 + vector],
                    selected);
                #pragma unroll
                for (int element = 0; element < 8; ++element) {
                    corrected[element] +=
                        selected[element] * scale * selected_probs[rank];
                }
            }
        }
        dE[output_offset] = pack_bfloat16x8(corrected);
    }
}

__global__ void sparse_correct_scaled_dE_vec2_kernel(
    __nv_bfloat162* __restrict__ dE,
    __nv_bfloat162* __restrict__ dC,
    const __nv_bfloat162* __restrict__ x,
    const __nv_bfloat162* __restrict__ weight,
    const int64_t* __restrict__ targets,
    const float* __restrict__ scale_ptr,
    int64_t M,
    int64_t K2,
    int64_t ignore_index
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = M * K2;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / K2;
    const int64_t pair = idx - row * K2;
    const int64_t target = targets[row];
    const float scale = *scale_ptr;

    const float2 dE_raw = __bfloat1622float2(dE[idx]);
    float2 dE_val = {dE_raw.x * scale, dE_raw.y * scale};
    if (target != ignore_index) {
        const float2 w_val = __bfloat1622float2(weight[target * K2 + pair]);
        dE_val.x -= w_val.x * scale;
        dE_val.y -= w_val.y * scale;
    }
    dE[idx] = __floats2bfloat162_rn(dE_val.x, dE_val.y);
}

template <int ENTRIES_PER_BLOCK>
__global__ void grouped_sparse_correct_dC_kernel(
    __nv_bfloat16* __restrict__ dC,
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ sorted_vocab_rows,
    const int* __restrict__ sorted_x_rows,
    const float* __restrict__ sorted_coefficients,
    int64_t entries,
    int64_t M,
    int64_t K,
    int64_t V
) {
    const int64_t block_start =
        static_cast<int64_t>(blockIdx.x) * ENTRIES_PER_BLOCK;
    __shared__ int64_t segment_end;
    #pragma unroll
    for (int candidate = 0; candidate < ENTRIES_PER_BLOCK; ++candidate) {
        const int64_t segment_start = block_start + candidate;
        if (segment_start >= entries) break;
        const int vocab_row = sorted_vocab_rows[segment_start];
        const bool is_segment_start =
            segment_start == 0 ||
            sorted_vocab_rows[segment_start - 1] != vocab_row;
        if (!is_segment_start || vocab_row < 0 || vocab_row >= V) continue;

        if (threadIdx.x == 0) {
            int64_t end = segment_start + 1;
            while (end < entries && sorted_vocab_rows[end] == vocab_row) {
                ++end;
            }
            segment_end = end;
        }
        __syncthreads();

        for (int64_t col = threadIdx.x; col < K; col += blockDim.x) {
            float correction = 0.0f;
            for (int64_t entry = segment_start; entry < segment_end; ++entry) {
                const int x_row = sorted_x_rows[entry];
                if (x_row >= 0 && x_row < M) {
                    correction = fmaf(
                        __bfloat162float(
                            x[static_cast<int64_t>(x_row) * K + col]),
                        sorted_coefficients[entry],
                        correction);
                }
            }
            const int64_t output_idx =
                static_cast<int64_t>(vocab_row) * K + col;
            dC[output_idx] = __float2bfloat16(
                __bfloat162float(dC[output_idx]) + correction);
        }
        __syncthreads();
    }
}

template <int ENTRIES_PER_BLOCK>
__global__ void grouped_sparse_correct_dC_vec2_kernel(
    __nv_bfloat162* __restrict__ dC,
    const __nv_bfloat162* __restrict__ x,
    const int* __restrict__ sorted_vocab_rows,
    const int* __restrict__ sorted_x_rows,
    const float* __restrict__ sorted_coefficients,
    int64_t entries,
    int64_t M,
    int64_t K2,
    int64_t V
) {
    const int64_t block_start =
        static_cast<int64_t>(blockIdx.x) * ENTRIES_PER_BLOCK;
    __shared__ int64_t segment_end;
    #pragma unroll
    for (int candidate = 0; candidate < ENTRIES_PER_BLOCK; ++candidate) {
        const int64_t segment_start = block_start + candidate;
        if (segment_start >= entries) break;
        const int vocab_row = sorted_vocab_rows[segment_start];
        const bool is_segment_start =
            segment_start == 0 ||
            sorted_vocab_rows[segment_start - 1] != vocab_row;
        if (!is_segment_start || vocab_row < 0 || vocab_row >= V) continue;

        if (threadIdx.x == 0) {
            int64_t end = segment_start + 1;
            while (end < entries && sorted_vocab_rows[end] == vocab_row) {
                ++end;
            }
            segment_end = end;
        }
        __syncthreads();

        for (int64_t pair = threadIdx.x; pair < K2; pair += blockDim.x) {
            float2 correction = {0.0f, 0.0f};
            for (int64_t entry = segment_start; entry < segment_end; ++entry) {
                const int x_row = sorted_x_rows[entry];
                if (x_row >= 0 && x_row < M) {
                    const float2 x_val = __bfloat1622float2(
                        x[static_cast<int64_t>(x_row) * K2 + pair]);
                    const float coefficient = sorted_coefficients[entry];
                    correction.x = fmaf(x_val.x, coefficient, correction.x);
                    correction.y = fmaf(x_val.y, coefficient, correction.y);
                }
            }
            const int64_t output_idx =
                static_cast<int64_t>(vocab_row) * K2 + pair;
            const float2 base = __bfloat1622float2(dC[output_idx]);
            dC[output_idx] = __floats2bfloat162_rn(
                base.x + correction.x,
                base.y + correction.y);
        }
        __syncthreads();
    }
}

template <int ENTRIES_PER_BLOCK>
__global__ void grouped_sparse_correct_dC_vec8_kernel(
    BFloat16x8* __restrict__ dC,
    const BFloat16x8* __restrict__ x,
    const int* __restrict__ sorted_vocab_rows,
    const int* __restrict__ sorted_x_rows,
    const float* __restrict__ sorted_coefficients,
    int64_t entries,
    int64_t M,
    int64_t K8,
    int64_t V
) {
    const int64_t block_start =
        static_cast<int64_t>(blockIdx.x) * ENTRIES_PER_BLOCK;
    __shared__ int64_t segment_end;
    #pragma unroll
    for (int candidate = 0; candidate < ENTRIES_PER_BLOCK; ++candidate) {
        const int64_t segment_start = block_start + candidate;
        if (segment_start >= entries) break;
        const int vocab_row = sorted_vocab_rows[segment_start];
        const bool is_segment_start =
            segment_start == 0 ||
            sorted_vocab_rows[segment_start - 1] != vocab_row;
        if (!is_segment_start || vocab_row < 0 || vocab_row >= V) continue;

        if (threadIdx.x == 0) {
            int64_t end = segment_start + 1;
            while (end < entries && sorted_vocab_rows[end] == vocab_row) {
                ++end;
            }
            segment_end = end;
        }
        __syncthreads();

        for (int64_t vector = threadIdx.x; vector < K8;
             vector += blockDim.x) {
            float correction[8] = {
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f};
            for (int64_t entry = segment_start; entry < segment_end; ++entry) {
                const int x_row = sorted_x_rows[entry];
                if (x_row >= 0 && x_row < M) {
                    float x_values[8];
                    unpack_bfloat16x8(
                        x[static_cast<int64_t>(x_row) * K8 + vector],
                        x_values);
                    const float coefficient = sorted_coefficients[entry];
                    #pragma unroll
                    for (int element = 0; element < 8; ++element) {
                        correction[element] = fmaf(
                            x_values[element], coefficient,
                            correction[element]);
                    }
                }
            }
            const int64_t output_offset =
                static_cast<int64_t>(vocab_row) * K8 + vector;
            float output[8];
            unpack_bfloat16x8(dC[output_offset], output);
            #pragma unroll
            for (int element = 0; element < 8; ++element) {
                output[element] += correction[element];
            }
            dC[output_offset] = pack_bfloat16x8(output);
        }
        __syncthreads();
    }
}

__global__ void mark_compact_target_topk_rows_kernel(
    const int64_t* __restrict__ targets,
    const int* __restrict__ topk_indices,
    int* __restrict__ vocab_to_slot,
    int64_t M,
    int64_t V,
    int topk_split,
    int64_t ignore_index
) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const int64_t target = targets[row];
    if (target == ignore_index) return;
    if (target >= 0 && target < V) {
        atomicCAS(vocab_to_slot + target, -1, -2);
    }
    for (int rank = 0; rank < topk_split; ++rank) {
        const int selected = topk_indices[row * topk_split + rank];
        if (selected >= 0 && selected < V) {
            atomicCAS(vocab_to_slot + selected, -1, -2);
        }
    }
}

__global__ void enumerate_compact_rows_kernel(
    int* __restrict__ vocab_to_slot,
    int* __restrict__ slot_to_vocab,
    int* __restrict__ slot_count,
    int64_t V
) {
    const int64_t vocab =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vocab >= V || vocab_to_slot[vocab] != -2) return;
    const int slot = atomicAdd(slot_count, 1);
    vocab_to_slot[vocab] = slot;
    slot_to_vocab[slot] = static_cast<int>(vocab);
}

__global__ void fill_compact_target_topk_coefficients_kernel(
    __nv_bfloat16* __restrict__ coefficient_matrix,
    const int* __restrict__ vocab_to_slot,
    const int64_t* __restrict__ targets,
    const float* __restrict__ target_probs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_probs,
    int64_t M,
    int64_t V,
    int topk_split,
    int64_t ignore_index,
    int slot_begin,
    int slot_end
) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M) return;
    const int64_t target = targets[row];
    if (target == ignore_index) return;
    if (target >= 0 && target < V) {
        const int slot = vocab_to_slot[target];
        if (slot >= slot_begin && slot < slot_end) {
            coefficient_matrix[
                static_cast<int64_t>(slot - slot_begin) * M + row] =
                __float2bfloat16(-(1.0f - target_probs[row]));
        }
    }
    for (int rank = 0; rank < topk_split; ++rank) {
        const int64_t offset = row * topk_split + rank;
        const int selected = topk_indices[offset];
        if (selected >= 0 && selected < V) {
            const int slot = vocab_to_slot[selected];
            if (slot >= slot_begin && slot < slot_end) {
                coefficient_matrix[
                    static_cast<int64_t>(slot - slot_begin) * M + row] =
                    __float2bfloat16(topk_probs[offset]);
            }
        }
    }
}

int validate_compact_target_topk_inputs(
    const torch::Tensor& targets,
    const torch::Tensor& target_probs,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_probs,
    int64_t vocab_size
) {
    TORCH_CHECK(
        targets.is_cuda() && target_probs.is_cuda() &&
            topk_indices.is_cuda() && topk_probs.is_cuda());
    TORCH_CHECK(
        targets.is_contiguous() && target_probs.is_contiguous() &&
            topk_indices.is_contiguous() && topk_probs.is_contiguous());
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(topk_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(targets.dim() == 1 && target_probs.sizes() == targets.sizes());
    TORCH_CHECK(topk_indices.dim() == 1 || topk_indices.dim() == 2);
    TORCH_CHECK(topk_probs.sizes() == topk_indices.sizes());
    TORCH_CHECK(topk_indices.size(0) == targets.size(0));
    TORCH_CHECK(vocab_size > 0);
    TORCH_CHECK(targets.numel() > 0);
    return topk_indices.dim() == 1
        ? 1
        : static_cast<int>(topk_indices.size(1));
}

struct CompactVocabularyAssignment {
    torch::Tensor vocab_to_slot;
    torch::Tensor slot_to_vocab;
    int group_count;
};

CompactVocabularyAssignment build_compact_vocabulary_assignment(
    const torch::Tensor& targets,
    const torch::Tensor& topk_indices,
    int64_t vocab_size,
    int topk_split,
    int64_t ignore_index
) {
    const int64_t M = targets.numel();
    const int threads = 256;
    const int row_blocks = static_cast<int>((M + threads - 1) / threads);
    const int vocab_blocks = static_cast<int>(
        (vocab_size + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto int_options = targets.options().dtype(torch::kInt32);
    auto vocab_to_slot = torch::empty({vocab_size}, int_options);
    auto slot_to_vocab = torch::empty(
        {std::min<int64_t>(vocab_size, M * (topk_split + 1LL))},
        int_options);
    auto slot_count = torch::empty({1}, int_options);
    TORCH_CHECK(
        cudaMemsetAsync(
            vocab_to_slot.data_ptr<int>(), 0xff,
            vocab_to_slot.numel() * sizeof(int), stream) == cudaSuccess,
        "failed to initialize compact vocabulary map");
    TORCH_CHECK(
        cudaMemsetAsync(slot_count.data_ptr<int>(), 0, sizeof(int), stream) ==
            cudaSuccess,
        "failed to initialize compact slot counter");

    mark_compact_target_topk_rows_kernel<<<row_blocks, threads, 0, stream>>>(
        targets.data_ptr<int64_t>(),
        topk_indices.data_ptr<int>(),
        vocab_to_slot.data_ptr<int>(),
        M,
        vocab_size,
        topk_split,
        ignore_index);
    enumerate_compact_rows_kernel<<<vocab_blocks, threads, 0, stream>>>(
        vocab_to_slot.data_ptr<int>(),
        slot_to_vocab.data_ptr<int>(),
        slot_count.data_ptr<int>(),
        vocab_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int group_count = 0;
    TORCH_CHECK(
        cudaMemcpyAsync(
            &group_count, slot_count.data_ptr<int>(), sizeof(int),
            cudaMemcpyDeviceToHost, stream) == cudaSuccess,
        "failed to read compact slot count");
    TORCH_CHECK(
        cudaStreamSynchronize(stream) == cudaSuccess,
        "failed to synchronize compact slot assignment");
    TORCH_CHECK(group_count >= 0 && group_count <= slot_to_vocab.numel());
    return {
        std::move(vocab_to_slot),
        std::move(slot_to_vocab),
        group_count,
    };
}

void fill_compact_target_topk_coefficients(
    torch::Tensor coefficient_matrix,
    const torch::Tensor& vocab_to_slot,
    const torch::Tensor& targets,
    const torch::Tensor& target_probs,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_probs,
    int64_t vocab_size,
    int topk_split,
    int64_t ignore_index,
    int slot_begin
) {
    if (coefficient_matrix.numel() == 0) return;
    const int64_t M = targets.numel();
    const int threads = 256;
    const int row_blocks = static_cast<int>((M + threads - 1) / threads);
    const int slot_end = slot_begin + coefficient_matrix.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fill_compact_target_topk_coefficients_kernel
        <<<row_blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(
                coefficient_matrix.data_ptr()),
            vocab_to_slot.data_ptr<int>(),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            topk_indices.data_ptr<int>(),
            topk_probs.data_ptr<float>(),
            M,
            vocab_size,
            topk_split,
            ignore_index,
            slot_begin,
            slot_end);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::vector<torch::Tensor> compact_target_topk_coefficients(
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor topk_indices,
    torch::Tensor topk_probs,
    int64_t vocab_size,
    int64_t ignore_index
) {
    const int topk_split = validate_compact_target_topk_inputs(
        targets, target_probs, topk_indices, topk_probs, vocab_size);
    const int64_t M = targets.numel();
    auto assignment = build_compact_vocabulary_assignment(
        targets, topk_indices, vocab_size, topk_split, ignore_index);
    auto coefficient_matrix = torch::zeros(
        {assignment.group_count, M},
        targets.options().dtype(torch::kBFloat16));
    fill_compact_target_topk_coefficients(
        coefficient_matrix,
        assignment.vocab_to_slot,
        targets,
        target_probs,
        topk_indices,
        topk_probs,
        vocab_size,
        topk_split,
        ignore_index,
        0);
    return {
        assignment.slot_to_vocab.narrow(0, 0, assignment.group_count),
        coefficient_matrix,
    };
}

std::vector<torch::Tensor> compact_target_topk_correction(
    torch::Tensor x,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor topk_indices,
    torch::Tensor topk_probs,
    int64_t vocab_size,
    int64_t ignore_index,
    int64_t chunk_rows
) {
    const int topk_split = validate_compact_target_topk_inputs(
        targets, target_probs, topk_indices, topk_probs, vocab_size);
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.dim() == 2);
    TORCH_CHECK(x.size(0) == targets.numel());
    TORCH_CHECK(x.device() == targets.device());
    TORCH_CHECK(chunk_rows > 0 && chunk_rows <= INT_MAX);

    const int64_t M = targets.numel();
    const int64_t K = x.size(1);
    auto assignment = build_compact_vocabulary_assignment(
        targets, topk_indices, vocab_size, topk_split, ignore_index);
    auto compact_correction = torch::empty(
        {assignment.group_count, K}, x.options());
    for (int64_t slot_begin = 0; slot_begin < assignment.group_count;
         slot_begin += chunk_rows) {
        const int64_t rows = std::min<int64_t>(
            chunk_rows, assignment.group_count - slot_begin);
        auto coefficient_chunk = torch::zeros({rows, M}, x.options());
        fill_compact_target_topk_coefficients(
            coefficient_chunk,
            assignment.vocab_to_slot,
            targets,
            target_probs,
            topk_indices,
            topk_probs,
            vocab_size,
            topk_split,
            ignore_index,
            static_cast<int>(slot_begin));
        auto correction_chunk = compact_correction.narrow(
            0, slot_begin, rows);
        at::mm_out(correction_chunk, coefficient_chunk, x);
    }
    return {
        assignment.slot_to_vocab.narrow(0, 0, assignment.group_count),
        compact_correction,
    };
}

void sparse_correct(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() && targets.is_cuda() && scale.is_cuda());
    TORCH_CHECK(dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() && weight.is_contiguous() && targets.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<false, 0><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            nullptr,
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index
        );
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<false, 0><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            nullptr,
            nullptr,
            nullptr,
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correct_target_split(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(
        dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() &&
            targets.is_cuda() && target_probs.is_cuda() && scale.is_cuda());
    TORCH_CHECK(
        dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() &&
            weight.is_contiguous() && targets.is_contiguous() &&
            target_probs.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(target_probs.numel() == x.size(0));
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<true, 0><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            nullptr,
            nullptr,
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index);
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<true, 0><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            nullptr,
            nullptr,
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correct_target_top1_split(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor top1_indices,
    torch::Tensor top1_probs,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(
        dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() &&
            targets.is_cuda() && target_probs.is_cuda() &&
            top1_indices.is_cuda() && top1_probs.is_cuda() &&
            scale.is_cuda());
    TORCH_CHECK(
        dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() &&
            weight.is_contiguous() && targets.is_contiguous() &&
            target_probs.is_contiguous() && top1_indices.is_contiguous() &&
            top1_probs.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(top1_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(top1_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(target_probs.numel() == x.size(0));
    TORCH_CHECK(top1_indices.numel() == x.size(0));
    TORCH_CHECK(top1_probs.numel() == x.size(0));
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<true, 1><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top1_indices.data_ptr<int>(),
            top1_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index);
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<true, 1><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top1_indices.data_ptr<int>(),
            top1_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correct_target_top2_split(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor top2_indices,
    torch::Tensor top2_probs,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(
        dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() &&
            targets.is_cuda() && target_probs.is_cuda() &&
            top2_indices.is_cuda() && top2_probs.is_cuda() &&
            scale.is_cuda());
    TORCH_CHECK(
        dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() &&
            weight.is_contiguous() && targets.is_contiguous() &&
            target_probs.is_contiguous() && top2_indices.is_contiguous() &&
            top2_probs.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(top2_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(top2_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(target_probs.numel() == x.size(0));
    TORCH_CHECK(
        top2_indices.dim() == 2 && top2_indices.size(0) == x.size(0) &&
            top2_indices.size(1) == 2);
    TORCH_CHECK(top2_probs.sizes() == top2_indices.sizes());
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<true, 2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top2_indices.data_ptr<int>(),
            top2_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index);
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<true, 2><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top2_indices.data_ptr<int>(),
            top2_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correct_target_top4_split(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor top4_indices,
    torch::Tensor top4_probs,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(
        dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() &&
            targets.is_cuda() && target_probs.is_cuda() &&
            top4_indices.is_cuda() && top4_probs.is_cuda() &&
            scale.is_cuda());
    TORCH_CHECK(
        dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() &&
            weight.is_contiguous() && targets.is_contiguous() &&
            target_probs.is_contiguous() && top4_indices.is_contiguous() &&
            top4_probs.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(top4_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(top4_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(target_probs.numel() == x.size(0));
    TORCH_CHECK(
        top4_indices.dim() == 2 && top4_indices.size(0) == x.size(0) &&
            top4_indices.size(1) == 4);
    TORCH_CHECK(top4_probs.sizes() == top4_indices.sizes());
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<true, 4><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top4_indices.data_ptr<int>(),
            top4_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index);
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<true, 4><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top4_indices.data_ptr<int>(),
            top4_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correct_target_top6_split(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor top6_indices,
    torch::Tensor top6_probs,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(
        dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() &&
            targets.is_cuda() && target_probs.is_cuda() &&
            top6_indices.is_cuda() && top6_probs.is_cuda() &&
            scale.is_cuda());
    TORCH_CHECK(
        dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() &&
            weight.is_contiguous() && targets.is_contiguous() &&
            target_probs.is_contiguous() && top6_indices.is_contiguous() &&
            top6_probs.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(top6_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(top6_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(target_probs.numel() == x.size(0));
    TORCH_CHECK(
        top6_indices.dim() == 2 && top6_indices.size(0) == x.size(0) &&
            top6_indices.size(1) == 6);
    TORCH_CHECK(top6_probs.sizes() == top6_indices.sizes());
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<true, 6><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top6_indices.data_ptr<int>(),
            top6_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index);
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<true, 6><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            target_probs.data_ptr<float>(),
            top6_indices.data_ptr<int>(),
            top6_probs.data_ptr<float>(),
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int TOPK_SPLIT>
void launch_sparse_correct_target_topk_dE(
    torch::Tensor dE,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor topk_indices,
    torch::Tensor topk_probs,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    const int64_t M = dE.size(0);
    const int64_t K = dE.size(1);
    const int threads = sparse_threads("FP4_CCE_V4_SPARSE_DE_THREADS", 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 8 == 0 && use_vec2 && use_sparse_vec8()) {
        sparse_correct_target_topk_dE_vec8_kernel<TOPK_SPLIT>
            <<<M, threads, 0, stream>>>(
                reinterpret_cast<BFloat16x8*>(dE.data_ptr()),
                reinterpret_cast<const BFloat16x8*>(weight.data_ptr()),
                targets.data_ptr<int64_t>(),
                target_probs.data_ptr<float>(),
                topk_indices.data_ptr<int>(),
                topk_probs.data_ptr<float>(),
                scale.data_ptr<float>(),
                M,
                K / 8,
                ignore_index);
    } else if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_vec2_kernel<true, TOPK_SPLIT>
            <<<blocks, threads, 0, stream>>>(
                reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
                reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
                reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
                targets.data_ptr<int64_t>(),
                target_probs.data_ptr<float>(),
                topk_indices.data_ptr<int>(),
                topk_probs.data_ptr<float>(),
                scale.data_ptr<float>(),
                M,
                K2,
                ignore_index);
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_kernel<true, TOPK_SPLIT>
            <<<blocks, threads, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
                targets.data_ptr<int64_t>(),
                target_probs.data_ptr<float>(),
                topk_indices.data_ptr<int>(),
                topk_probs.data_ptr<float>(),
                scale.data_ptr<float>(),
                M,
                K,
                ignore_index);
    }
}

void sparse_correct_target_topk_dE(
    torch::Tensor dE,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor target_probs,
    torch::Tensor topk_indices,
    torch::Tensor topk_probs,
    torch::Tensor scale,
    int64_t topk_split,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(
        dE.is_cuda() && weight.is_cuda() && targets.is_cuda() &&
            target_probs.is_cuda() && topk_indices.is_cuda() &&
            topk_probs.is_cuda() && scale.is_cuda());
    TORCH_CHECK(
        dE.is_contiguous() && weight.is_contiguous() &&
            targets.is_contiguous() && target_probs.is_contiguous() &&
            topk_indices.is_contiguous() && topk_probs.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(target_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(topk_probs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(dE.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.size(1) == weight.size(1));
    TORCH_CHECK(targets.numel() == dE.size(0));
    TORCH_CHECK(target_probs.numel() == dE.size(0));
    TORCH_CHECK(topk_split == 1 || topk_split == 2 ||
                topk_split == 4 || topk_split == 6 ||
                topk_split == 8 || topk_split == 12 ||
                topk_split == 16);
    TORCH_CHECK(
        (topk_split == 1 && topk_indices.dim() == 1) ||
        (topk_indices.dim() == 2 && topk_indices.size(1) == topk_split));
    TORCH_CHECK(topk_indices.size(0) == dE.size(0));
    TORCH_CHECK(topk_probs.sizes() == topk_indices.sizes());
    TORCH_CHECK(scale.numel() == 1);

    if (topk_split == 1) {
        launch_sparse_correct_target_topk_dE<1>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    } else if (topk_split == 2) {
        launch_sparse_correct_target_topk_dE<2>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    } else if (topk_split == 4) {
        launch_sparse_correct_target_topk_dE<4>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    } else if (topk_split == 6) {
        launch_sparse_correct_target_topk_dE<6>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    } else if (topk_split == 8) {
        launch_sparse_correct_target_topk_dE<8>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    } else if (topk_split == 12) {
        launch_sparse_correct_target_topk_dE<12>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    } else {
        launch_sparse_correct_target_topk_dE<16>(
            dE, weight, targets, target_probs, topk_indices, topk_probs,
            scale, ignore_index, use_vec2);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correct_scaled_dE(
    torch::Tensor dE,
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor targets,
    torch::Tensor scale,
    int64_t ignore_index,
    bool use_vec2
) {
    TORCH_CHECK(dE.is_cuda() && dC.is_cuda() && x.is_cuda() && weight.is_cuda() && targets.is_cuda() && scale.is_cuda());
    TORCH_CHECK(dE.is_contiguous() && dC.is_contiguous() && x.is_contiguous() && weight.is_contiguous() && targets.is_contiguous());
    TORCH_CHECK(dE.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(targets.scalar_type() == torch::kInt64);
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2);
    TORCH_CHECK(dE.sizes() == x.sizes());
    TORCH_CHECK(dC.sizes() == weight.sizes());
    TORCH_CHECK(targets.numel() == x.size(0));
    TORCH_CHECK(scale.numel() == 1);

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 2 == 0 && use_vec2) {
        const int64_t K2 = K / 2;
        const int64_t total = M * K2;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_scaled_dE_vec2_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat162*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            scale.data_ptr<float>(),
            M,
            K2,
            ignore_index
        );
    } else {
        const int64_t total = M * K;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        sparse_correct_scaled_dE_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(dE.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            targets.data_ptr<int64_t>(),
            scale.data_ptr<float>(),
            M,
            K,
            ignore_index
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void grouped_sparse_correct_dC(
    torch::Tensor dC,
    torch::Tensor x,
    torch::Tensor sorted_vocab_rows,
    torch::Tensor sorted_x_rows,
    torch::Tensor sorted_coefficients,
    bool use_vec2,
    int64_t entries_per_block,
    int64_t threads_per_block
) {
    TORCH_CHECK(
        dC.is_cuda() && x.is_cuda() && sorted_vocab_rows.is_cuda() &&
            sorted_x_rows.is_cuda() && sorted_coefficients.is_cuda());
    TORCH_CHECK(
        dC.is_contiguous() && x.is_contiguous() &&
            sorted_vocab_rows.is_contiguous() && sorted_x_rows.is_contiguous() &&
            sorted_coefficients.is_contiguous());
    TORCH_CHECK(dC.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(sorted_vocab_rows.scalar_type() == torch::kInt32);
    TORCH_CHECK(sorted_x_rows.scalar_type() == torch::kInt32);
    TORCH_CHECK(sorted_coefficients.scalar_type() == torch::kFloat32);
    TORCH_CHECK(dC.dim() == 2 && x.dim() == 2);
    TORCH_CHECK(dC.size(1) == x.size(1));
    TORCH_CHECK(sorted_vocab_rows.dim() == 1);
    TORCH_CHECK(sorted_x_rows.sizes() == sorted_vocab_rows.sizes());
    TORCH_CHECK(sorted_coefficients.sizes() == sorted_vocab_rows.sizes());
    TORCH_CHECK(
        entries_per_block == 1 || entries_per_block == 2 ||
        entries_per_block == 3 || entries_per_block == 4 ||
        entries_per_block == 5 || entries_per_block == 8);
    TORCH_CHECK(
        threads_per_block == 128 || threads_per_block == 256 ||
        threads_per_block == 512);

    const int64_t entries = sorted_vocab_rows.numel();
    if (entries == 0) {
        return;
    }
    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t V = dC.size(0);
    const int threads = static_cast<int>(threads_per_block);
    const int blocks = static_cast<int>(
        (entries + entries_per_block - 1) / entries_per_block);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K % 8 == 0 && use_vec2 && use_sparse_vec8()) {
        #define LAUNCH_GROUPED_VEC8(E) \
            grouped_sparse_correct_dC_vec8_kernel<E> \
                <<<blocks, threads, 0, stream>>>( \
                    reinterpret_cast<BFloat16x8*>(dC.data_ptr()), \
                    reinterpret_cast<const BFloat16x8*>(x.data_ptr()), \
                    sorted_vocab_rows.data_ptr<int>(), \
                    sorted_x_rows.data_ptr<int>(), \
                    sorted_coefficients.data_ptr<float>(), \
                    entries, M, K / 8, V)
        if (entries_per_block == 1) {
            LAUNCH_GROUPED_VEC8(1);
        } else if (entries_per_block == 2) {
            LAUNCH_GROUPED_VEC8(2);
        } else if (entries_per_block == 3) {
            LAUNCH_GROUPED_VEC8(3);
        } else if (entries_per_block == 4) {
            LAUNCH_GROUPED_VEC8(4);
        } else if (entries_per_block == 5) {
            LAUNCH_GROUPED_VEC8(5);
        } else {
            LAUNCH_GROUPED_VEC8(8);
        }
        #undef LAUNCH_GROUPED_VEC8
    } else if (K % 2 == 0 && use_vec2) {
        #define LAUNCH_GROUPED_VEC2(E) \
            grouped_sparse_correct_dC_vec2_kernel<E><<<blocks, threads, 0, stream>>>( \
                reinterpret_cast<__nv_bfloat162*>(dC.data_ptr()), \
                reinterpret_cast<const __nv_bfloat162*>(x.data_ptr()), \
                sorted_vocab_rows.data_ptr<int>(), \
                sorted_x_rows.data_ptr<int>(), \
                sorted_coefficients.data_ptr<float>(), \
                entries, M, K / 2, V)
        if (entries_per_block == 1) {
            LAUNCH_GROUPED_VEC2(1);
        } else if (entries_per_block == 2) {
            LAUNCH_GROUPED_VEC2(2);
        } else if (entries_per_block == 3) {
            LAUNCH_GROUPED_VEC2(3);
        } else if (entries_per_block == 4) {
            LAUNCH_GROUPED_VEC2(4);
        } else if (entries_per_block == 5) {
            LAUNCH_GROUPED_VEC2(5);
        } else {
            LAUNCH_GROUPED_VEC2(8);
        }
        #undef LAUNCH_GROUPED_VEC2
    } else {
        #define LAUNCH_GROUPED(E) \
            grouped_sparse_correct_dC_kernel<E><<<blocks, threads, 0, stream>>>( \
                reinterpret_cast<__nv_bfloat16*>(dC.data_ptr()), \
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), \
                sorted_vocab_rows.data_ptr<int>(), \
                sorted_x_rows.data_ptr<int>(), \
                sorted_coefficients.data_ptr<float>(), \
                entries, M, K, V)
        if (entries_per_block == 1) {
            LAUNCH_GROUPED(1);
        } else if (entries_per_block == 2) {
            LAUNCH_GROUPED(2);
        } else if (entries_per_block == 3) {
            LAUNCH_GROUPED(3);
        } else if (entries_per_block == 4) {
            LAUNCH_GROUPED(4);
        } else if (entries_per_block == 5) {
            LAUNCH_GROUPED(5);
        } else {
            LAUNCH_GROUPED(8);
        }
        #undef LAUNCH_GROUPED
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_correct", &sparse_correct, "FP4 CCE v4 sparse label correction");
    m.def(
        "sparse_correct_target_split",
        &sparse_correct_target_split,
        "FP4 CCE v4 sparse correction with exact target probability");
    m.def(
        "sparse_correct_target_top1_split",
        &sparse_correct_target_top1_split,
        "FP4 CCE v4 sparse correction with exact target and top-1 probabilities");
    m.def(
        "sparse_correct_target_top2_split",
        &sparse_correct_target_top2_split,
        "FP4 CCE v4 sparse correction with exact target and top-2 probabilities");
    m.def(
        "sparse_correct_target_top4_split",
        &sparse_correct_target_top4_split,
        "FP4 CCE v4 sparse correction with exact target and top-4 probabilities");
    m.def(
        "sparse_correct_target_top6_split",
        &sparse_correct_target_top6_split,
        "FP4 CCE v4 sparse correction with exact target and top-6 probabilities");
    m.def(
        "sparse_correct_target_topk_dE",
        &sparse_correct_target_topk_dE,
        "FP4 CCE v4 target/top-k activation-gradient correction");
    m.def("sparse_correct_scaled_dE", &sparse_correct_scaled_dE,
          "FP4 CCE v4 sparse label correction with fused dE scaling");
    m.def(
        "grouped_sparse_correct_dC",
        &grouped_sparse_correct_dC,
        "FP4 CCE v4 grouped FP32 sparse weight-gradient correction");
    m.def(
        "compact_target_topk_coefficients",
        &compact_target_topk_coefficients,
        "FP4 CCE v4 compact target/top-k coefficient producer");
    m.def(
        "compact_target_topk_correction",
        &compact_target_topk_correction,
        "FP4 CCE v4 memory-bounded compact target/top-k correction");
}
