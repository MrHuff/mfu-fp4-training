// MXFP4 v4 Quantize — PyBind11 dispatch
//
// Exports:
//   mxfp4_quantize_for_gemm(input) → (fp4, scales)
//   mxfp4_group_quantize_dim0(input, Ms) → list[(fp4, scales)]

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <vector>
#include <algorithm>
#include <array>
#include <climits>
#include <dlfcn.h>
#include <cstdlib>

#include "mxfp4_v3_quantize.cuh"
#include "../../ThunderKittens/kernels/gemm/common/c1_rms_reduce.cuh"

using namespace mxfp4_v3;

static bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

constexpr unsigned long long MXFP4_FORWARD_SR_INVOCATION_STRIDE = 1ull << 40;
__device__ unsigned long long mxfp4_forward_sr_invocation_offset = 0;

__global__ void prepare_mxfp4_forward_advancing_rng_state_kernel(
    unsigned long long* rng_state,
    unsigned long long rng_seed,
    unsigned long long rng_subsequence_base
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned long long offset = atomicAdd(
            &mxfp4_forward_sr_invocation_offset,
            MXFP4_FORWARD_SR_INVOCATION_STRIDE);
        rng_state[0] = rng_seed;
        rng_state[1] = rng_subsequence_base + offset;
    }
}

static torch::Tensor make_mxfp4_forward_advancing_rng_state(
    const torch::Tensor& input,
    uint64_t rng_seed,
    uint64_t rng_subsequence_base,
    cudaStream_t stream
) {
    auto rng_state = torch::empty(
        {2}, torch::dtype(torch::kInt64).device(input.device()));
    prepare_mxfp4_forward_advancing_rng_state_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(rng_state.data_ptr<int64_t>()),
        static_cast<unsigned long long>(rng_seed),
        static_cast<unsigned long long>(rng_subsequence_base));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rng_state;
}

static void create_tma_2d(
    CUtensorMap& tma,
    void* ptr,
    uint64_t dimY,
    uint64_t dimX,
    uint32_t boxY,
    uint32_t boxX,
    uint64_t strideX,
    size_t elemBits,
    CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE);

static torch::Tensor make_i64_device_tensor(
    const std::vector<int64_t>& values,
    torch::Device device,
    cudaStream_t stream
) {
    auto tensor = torch::empty(
        {static_cast<int64_t>(values.size())},
        torch::dtype(torch::kInt64).device(device));
    if (!values.empty()) {
        cudaMemcpyAsync(
            tensor.data_ptr<int64_t>(),
            values.data(),
            values.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice,
            stream);
    }
    return tensor;
}

__global__ void mxfp4_pack_grouped_rows_bf16_kernel(
    const IType* __restrict__ src,
    IType* __restrict__ dst,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ padded_starts,
    const int64_t* __restrict__ padded_rows,
    int64_t input_cols,
    int64_t output_cols,
    int64_t max_padded_rows
) {
    const int64_t expert = blockIdx.x;
    const int64_t padded = padded_rows[expert];
    const int64_t live = rows[expert];
    const int64_t src_start = starts[expert];
    const int64_t dst_start = padded_starts[expert];
    const int64_t total = padded * output_cols;
    for (int64_t linear = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.y * blockDim.x) {
        const int64_t row = linear / output_cols;
        const int64_t col = linear - row * output_cols;
        IType value = __float2bfloat16(0.0f);
        if (row < live && col < input_cols) {
            value = src[(src_start + row) * input_cols + col];
        }
        dst[(dst_start + row) * output_cols + col] = value;
    }
}

__global__ void mxfp4_pack_grouped_rows_bf16_vec_kernel(
    const int4* __restrict__ src,
    int4* __restrict__ dst,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ padded_starts,
    const int64_t* __restrict__ padded_rows,
    int64_t input_cols_vec,
    int64_t output_cols_vec,
    int64_t max_padded_rows
) {
    const int64_t expert = blockIdx.x;
    const int64_t padded = padded_rows[expert];
    const int64_t live = rows[expert];
    const int64_t src_start = starts[expert];
    const int64_t dst_start = padded_starts[expert];
    const int64_t total = padded * output_cols_vec;
    const int4 zero = {0, 0, 0, 0};
    for (int64_t linear = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.y * blockDim.x) {
        const int64_t row = linear / output_cols_vec;
        const int64_t col = linear - row * output_cols_vec;
        int4 value = zero;
        if (row < live && col < input_cols_vec) {
            value = src[(src_start + row) * input_cols_vec + col];
        }
        dst[(dst_start + row) * output_cols_vec + col] = value;
    }
}

__global__ void mxfp4_pack_indexed_rows_bf16_kernel(
    const IType* __restrict__ src,
    const int64_t* __restrict__ token_indices,
    IType* __restrict__ dst,
    int64_t num_batches,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t input_cols,
    int64_t output_cols
) {
    const int64_t total_padded = num_batches * padded_rows_per_batch;
    const int64_t total = total_padded * output_cols;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = linear / output_cols;
        const int64_t col = linear - row * output_cols;
        const int64_t batch = row / padded_rows_per_batch;
        const int64_t row_in_batch = row - batch * padded_rows_per_batch;
        IType value = __float2bfloat16(0.0f);
        if (batch < num_batches && row_in_batch < live_rows_per_batch && col < input_cols) {
            const int64_t routed_row = batch * live_rows_per_batch + row_in_batch;
            const int64_t src_row = token_indices[routed_row];
            value = src[src_row * input_cols + col];
        }
        dst[row * output_cols + col] = value;
    }
}

__global__ void mxfp4_pack_indexed_rows_bf16_grouped_kernel(
    const IType* __restrict__ src,
    const int64_t* __restrict__ token_indices,
    IType* __restrict__ dst,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t input_cols,
    int64_t output_cols
) {
    const int64_t expert = blockIdx.x;
    const int64_t total = padded_rows_per_batch * output_cols;
    const int64_t src_route_base = expert * live_rows_per_batch;
    const int64_t dst_base = expert * padded_rows_per_batch * output_cols;
    for (int64_t linear = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.y) * blockDim.x) {
        const int64_t row = linear / output_cols;
        const int64_t col = linear - row * output_cols;
        IType value = __float2bfloat16(0.0f);
        if (row < live_rows_per_batch && col < input_cols) {
            const int64_t src_row = token_indices[src_route_base + row];
            value = src[src_row * input_cols + col];
        }
        dst[dst_base + row * output_cols + col] = value;
    }
}

__global__ void mxfp4_pack_indexed_rows_bf16_vec_kernel(
    const int4* __restrict__ src,
    const int64_t* __restrict__ token_indices,
    int4* __restrict__ dst,
    int64_t num_batches,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t input_cols_vec,
    int64_t output_cols_vec
) {
    const int64_t total_padded = num_batches * padded_rows_per_batch;
    const int64_t total = total_padded * output_cols_vec;
    const int4 zero = {0, 0, 0, 0};
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = linear / output_cols_vec;
        const int64_t col = linear - row * output_cols_vec;
        const int64_t batch = row / padded_rows_per_batch;
        const int64_t row_in_batch = row - batch * padded_rows_per_batch;
        int4 value = zero;
        if (batch < num_batches && row_in_batch < live_rows_per_batch && col < input_cols_vec) {
            const int64_t routed_row = batch * live_rows_per_batch + row_in_batch;
            const int64_t src_row = token_indices[routed_row];
            value = src[src_row * input_cols_vec + col];
        }
        dst[row * output_cols_vec + col] = value;
    }
}

__global__ void mxfp4_pack_indexed_rows_bf16_grouped_vec_kernel(
    const int4* __restrict__ src,
    const int64_t* __restrict__ token_indices,
    int4* __restrict__ dst,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t input_cols_vec,
    int64_t output_cols_vec
) {
    const int64_t expert = blockIdx.x;
    const int64_t total = padded_rows_per_batch * output_cols_vec;
    const int64_t src_route_base = expert * live_rows_per_batch;
    const int64_t dst_base = expert * padded_rows_per_batch * output_cols_vec;
    const int4 zero = {0, 0, 0, 0};
    for (int64_t linear = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.y) * blockDim.x) {
        const int64_t row = linear / output_cols_vec;
        const int64_t col = linear - row * output_cols_vec;
        int4 value = zero;
        if (row < live_rows_per_batch && col < input_cols_vec) {
            const int64_t src_row = token_indices[src_route_base + row];
            value = src[src_row * input_cols_vec + col];
        }
        dst[dst_base + row * output_cols_vec + col] = value;
    }
}

__global__ void mxfp4_scatter_grouped_rows_bf16_kernel(
    const IType* __restrict__ src,
    IType* __restrict__ dst,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ padded_starts,
    int64_t input_cols,
    int64_t output_cols,
    int64_t max_live_rows
) {
    const int64_t expert = blockIdx.x;
    const int64_t live = rows[expert];
    const int64_t src_start = padded_starts[expert];
    const int64_t dst_start = starts[expert];
    const int64_t total = live * output_cols;
    for (int64_t linear = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.y * blockDim.x) {
        const int64_t row = linear / output_cols;
        const int64_t col = linear - row * output_cols;
        IType value = __float2bfloat16(0.0f);
        if (col < input_cols) {
            value = src[(src_start + row) * input_cols + col];
        }
        dst[(dst_start + row) * output_cols + col] = value;
    }
}

__global__ void mxfp4_scatter_grouped_rows_bf16_vec_kernel(
    const int4* __restrict__ src,
    int4* __restrict__ dst,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ rows,
    const int64_t* __restrict__ padded_starts,
    int64_t input_cols_vec,
    int64_t output_cols_vec,
    int64_t max_live_rows
) {
    const int64_t expert = blockIdx.x;
    const int64_t live = rows[expert];
    const int64_t src_start = padded_starts[expert];
    const int64_t dst_start = starts[expert];
    const int64_t total = live * output_cols_vec;
    for (int64_t linear = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.y * blockDim.x) {
        const int64_t row = linear / output_cols_vec;
        const int64_t col = linear - row * output_cols_vec;
        dst[(dst_start + row) * output_cols_vec + col] =
            src[(src_start + row) * input_cols_vec + col];
    }
}

__global__ void mxfp4_pack_w13_bf16_kernel(
    const IType* __restrict__ w1,
    const IType* __restrict__ w3,
    IType* __restrict__ out,
    int64_t E,
    int64_t H,
    int64_t D,
    int64_t H13n,
    int64_t Dk
) {
    const int64_t total = E * H13n * Dk;
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t col = linear % Dk;
        const int64_t row = linear / Dk;
        const int64_t h13 = row % H13n;
        const int64_t expert = row / H13n;
        IType value = __float2bfloat16(0.0f);
        if (col < D) {
            if (h13 < H) {
                value = w1[(expert * H + h13) * D + col];
            } else if (h13 < 2 * H) {
                value = w3[(expert * H + (h13 - H)) * D + col];
            }
        }
        out[linear] = value;
    }
}

__global__ void mxfp4_pack_w13_bf16_vec_kernel(
    const int4* __restrict__ w1,
    const int4* __restrict__ w3,
    int4* __restrict__ out,
    int64_t E,
    int64_t H,
    int64_t D_vec,
    int64_t H13n,
    int64_t Dk_vec
) {
    const int64_t total = E * H13n * Dk_vec;
    const int4 zero = {0, 0, 0, 0};
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t col = linear % Dk_vec;
        const int64_t row = linear / Dk_vec;
        const int64_t h13 = row % H13n;
        const int64_t expert = row / H13n;
        int4 value = zero;
        if (col < D_vec) {
            if (h13 < H) {
                value = w1[(expert * H + h13) * D_vec + col];
            } else if (h13 < 2 * H) {
                value = w3[(expert * H + (h13 - H)) * D_vec + col];
            }
        }
        out[linear] = value;
    }
}

__global__ void mxfp4_split_w13_bf16_vec_kernel(
    const int4* __restrict__ grad_w13,
    int4* __restrict__ grad_w1,
    int4* __restrict__ grad_w3,
    int64_t E,
    int64_t H,
    int64_t D_vec,
    int64_t H13n,
    int64_t Dn_vec
) {
    const int64_t total = E * H * D_vec;
    for (int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += (int64_t)gridDim.x * blockDim.x) {
        const int64_t col = linear % D_vec;
        const int64_t tmp = linear / D_vec;
        const int64_t h = tmp % H;
        const int64_t expert = tmp / H;
        grad_w1[linear] = grad_w13[(expert * H13n + h) * Dn_vec + col];
        grad_w3[linear] = grad_w13[(expert * H13n + H + h) * Dn_vec + col];
    }
}

torch::Tensor mxfp4_pack_w13_bf16(
    torch::Tensor w1,
    torch::Tensor w3,
    int64_t H13n,
    int64_t Dk
) {
    TORCH_CHECK(w1.is_cuda() && w1.is_contiguous());
    TORCH_CHECK(w3.is_cuda() && w3.is_contiguous());
    TORCH_CHECK(w1.scalar_type() == torch::kBFloat16 && w1.dim() == 3);
    TORCH_CHECK(w3.scalar_type() == torch::kBFloat16 && w3.dim() == 3);
    TORCH_CHECK(w1.sizes() == w3.sizes(), "w1 and w3 must have the same shape");
    const int64_t E = w1.size(0);
    const int64_t H = w1.size(1);
    const int64_t D = w1.size(2);
    TORCH_CHECK(H13n >= 2 * H && Dk >= D, "output padding must cover packed w1/w3 shape");
    auto out = torch::empty(
        {E * H13n, Dk},
        torch::dtype(torch::kBFloat16).device(w1.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    if (D % 8 == 0 && Dk % 8 == 0) {
        const int64_t D_vec = D / 8;
        const int64_t Dk_vec = Dk / 8;
        const int64_t total = E * H13n * Dk_vec;
        const int blocks = static_cast<int>(std::min<int64_t>(65535, (total + threads - 1) / threads));
        mxfp4_pack_w13_bf16_vec_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const int4*>(w1.data_ptr()),
            reinterpret_cast<const int4*>(w3.data_ptr()),
            reinterpret_cast<int4*>(out.data_ptr()),
            E,
            H,
            D_vec,
            H13n,
            Dk_vec);
    } else {
        const int64_t total = E * H13n * Dk;
        const int blocks = static_cast<int>(std::min<int64_t>(65535, (total + threads - 1) / threads));
        mxfp4_pack_w13_bf16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const IType*>(w1.data_ptr()),
            reinterpret_cast<const IType*>(w3.data_ptr()),
            reinterpret_cast<IType*>(out.data_ptr()),
            E,
            H,
            D,
            H13n,
            Dk);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 pack w13 bf16: ", cudaGetErrorString(err));
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> mxfp4_split_w13_bf16(
    torch::Tensor grad_w13,
    int64_t H,
    int64_t D
) {
    TORCH_CHECK(grad_w13.is_cuda() && grad_w13.is_contiguous());
    TORCH_CHECK(grad_w13.scalar_type() == torch::kBFloat16 && grad_w13.dim() == 3);
    const int64_t E = grad_w13.size(0);
    const int64_t H13n = grad_w13.size(1);
    const int64_t Dn = grad_w13.size(2);
    TORCH_CHECK(H13n >= 2 * H && Dn >= D, "grad_w13 does not cover requested w1/w3 slices");
    TORCH_CHECK(D % 8 == 0 && Dn % 8 == 0, "D and Dn must be divisible by 8 for vectorized split");
    auto grad_w1 = torch::empty(
        {E, H, D},
        torch::dtype(torch::kBFloat16).device(grad_w13.device()));
    auto grad_w3 = torch::empty(
        {E, H, D},
        torch::dtype(torch::kBFloat16).device(grad_w13.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    const int64_t D_vec = D / 8;
    const int64_t Dn_vec = Dn / 8;
    const int64_t total = E * H * D_vec;
    const int blocks = static_cast<int>(std::min<int64_t>(65535, (total + threads - 1) / threads));
    mxfp4_split_w13_bf16_vec_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(grad_w13.data_ptr()),
        reinterpret_cast<int4*>(grad_w1.data_ptr()),
        reinterpret_cast<int4*>(grad_w3.data_ptr()),
        E,
        H,
        D_vec,
        H13n,
        Dn_vec);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split w13 bf16: ", cudaGetErrorString(err));
    return std::make_tuple(grad_w1, grad_w3);
}

torch::Tensor mxfp4_pack_grouped_rows_bf16(
    torch::Tensor input,
    std::vector<int64_t> starts,
    std::vector<int64_t> rows,
    std::vector<int64_t> padded_rows,
    int64_t output_cols
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(starts.size() == rows.size() && rows.size() == padded_rows.size(),
                "starts, rows, and padded_rows length mismatch");
    TORCH_CHECK(output_cols > 0, "output_cols must be positive");
    const int64_t n = static_cast<int64_t>(rows.size());
    TORCH_CHECK(n > 0, "expected at least one group");
    const int64_t input_cols = input.size(1);
    int64_t total_padded = 0;
    int64_t max_padded = 0;
    std::vector<int64_t> padded_starts;
    padded_starts.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        TORCH_CHECK(rows[i] >= 0 && padded_rows[i] >= rows[i], "invalid grouped row counts");
        TORCH_CHECK(starts[i] >= 0 && starts[i] + rows[i] <= input.size(0), "group row range out of input bounds");
        padded_starts.push_back(total_padded);
        total_padded += padded_rows[i];
        max_padded = std::max(max_padded, padded_rows[i]);
    }
    auto output = torch::empty(
        {total_padded, output_cols},
        torch::dtype(torch::kBFloat16).device(input.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto starts_dev = make_i64_device_tensor(starts, input.device(), stream);
    auto rows_dev = make_i64_device_tensor(rows, input.device(), stream);
    auto padded_starts_dev = make_i64_device_tensor(padded_starts, input.device(), stream);
    auto padded_rows_dev = make_i64_device_tensor(padded_rows, input.device(), stream);
    constexpr int threads = 256;
    const bool use_vec = (input_cols % 8 == 0) && (output_cols % 8 == 0);
    const int64_t input_cols_work = use_vec ? input_cols / 8 : input_cols;
    const int64_t output_cols_work = use_vec ? output_cols / 8 : output_cols;
    const int64_t work_per_group = std::max<int64_t>(1, max_padded * output_cols_work);
    const int grid_y = static_cast<int>(std::min<int64_t>(1024, (work_per_group + threads - 1) / threads));
    dim3 grid(n, grid_y);
    if (use_vec) {
        mxfp4_pack_grouped_rows_bf16_vec_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const int4*>(input.data_ptr()),
            reinterpret_cast<int4*>(output.data_ptr()),
            starts_dev.data_ptr<int64_t>(),
            rows_dev.data_ptr<int64_t>(),
            padded_starts_dev.data_ptr<int64_t>(),
            padded_rows_dev.data_ptr<int64_t>(),
            input_cols_work,
            output_cols_work,
            max_padded);
    } else {
        mxfp4_pack_grouped_rows_bf16_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const IType*>(input.data_ptr()),
            reinterpret_cast<IType*>(output.data_ptr()),
            starts_dev.data_ptr<int64_t>(),
            rows_dev.data_ptr<int64_t>(),
            padded_starts_dev.data_ptr<int64_t>(),
            padded_rows_dev.data_ptr<int64_t>(),
            input_cols,
            output_cols,
            max_padded);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 grouped bf16 row pack: ", cudaGetErrorString(err));
    return output;
}

torch::Tensor mxfp4_pack_indexed_rows_bf16(
    torch::Tensor input,
    torch::Tensor token_indices,
    int64_t num_batches,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t output_cols
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(token_indices.is_cuda() && token_indices.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt64 && token_indices.dim() == 1);
    TORCH_CHECK(num_batches > 0, "num_batches must be positive");
    TORCH_CHECK(live_rows_per_batch > 0, "live_rows_per_batch must be positive");
    TORCH_CHECK(padded_rows_per_batch >= live_rows_per_batch,
                "padded rows must cover live rows");
    TORCH_CHECK(token_indices.size(0) >= num_batches * live_rows_per_batch,
                "token_indices does not contain the requested rows");
    TORCH_CHECK(output_cols >= input.size(1), "output_cols must cover input columns");
    const int64_t input_cols = input.size(1);
    const int64_t total_padded = num_batches * padded_rows_per_batch;
    auto output = torch::empty(
        {total_padded, output_cols},
        torch::dtype(torch::kBFloat16).device(input.device()));
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    const bool use_vec = (input_cols % 8 == 0) && (output_cols % 8 == 0);
    if (use_vec) {
        const int64_t input_cols_vec = input_cols / 8;
        const int64_t output_cols_vec = output_cols / 8;
        const int64_t work_per_group = padded_rows_per_batch * output_cols_vec;
        const int grid_y = static_cast<int>(std::min<int64_t>(1024, (work_per_group + threads - 1) / threads));
        dim3 grid(num_batches, grid_y);
        mxfp4_pack_indexed_rows_bf16_grouped_vec_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const int4*>(input.data_ptr()),
            token_indices.data_ptr<int64_t>(),
            reinterpret_cast<int4*>(output.data_ptr()),
            live_rows_per_batch,
            padded_rows_per_batch,
            input_cols_vec,
            output_cols_vec);
    } else {
        const int64_t work_per_group = padded_rows_per_batch * output_cols;
        const int grid_y = static_cast<int>(std::min<int64_t>(1024, (work_per_group + threads - 1) / threads));
        dim3 grid(num_batches, grid_y);
        mxfp4_pack_indexed_rows_bf16_grouped_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const IType*>(input.data_ptr()),
            token_indices.data_ptr<int64_t>(),
            reinterpret_cast<IType*>(output.data_ptr()),
            live_rows_per_batch,
            padded_rows_per_batch,
            input_cols,
            output_cols);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 indexed bf16 row pack: ", cudaGetErrorString(err));
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_pack_grouped_rows_quantize_row_and_col(
    torch::Tensor input,
    int64_t num_batches,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t output_cols,
    int mode
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(num_batches > 0, "num_batches must be positive");
    TORCH_CHECK(live_rows_per_batch > 0, "live_rows_per_batch must be positive");
    TORCH_CHECK(padded_rows_per_batch >= live_rows_per_batch,
                "padded rows must cover live rows");
    TORCH_CHECK(input.size(0) >= num_batches * live_rows_per_batch,
                "input does not contain the requested live grouped rows");
    TORCH_CHECK(output_cols >= input.size(1), "output_cols must be at least input cols");
    TORCH_CHECK(input.size(1) % 128 == 0, "input cols must be 128-aligned for TMA fast path");
    TORCH_CHECK(output_cols % 128 == 0 && padded_rows_per_batch % 128 == 0,
                "packed grouped quant requires 128-aligned output dims");
    auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(set_device_err));

    const int64_t total_padded = num_batches * padded_rows_per_batch;
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty(
        {total_padded, output_cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {total_padded / 128, output_cols / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {output_cols, total_padded / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {output_cols / 128, total_padded / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), num_batches * live_rows_per_batch, input.size(1),
                  TILE_DIM, TILE_DIM, input.size(1), 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), total_padded, output_cols,
                  TILE_DIM, TILE_DIM, output_cols, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), output_cols, total_padded,
                  TILE_DIM, TILE_DIM, total_padded, 4);

    const int dshmem = v4_rowcol_shmem_size();
    cudaFuncSetAttribute(mxfp4_v4_grouped_rows_pack_rowcol_kernel<QuantMode::RTE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaFuncSetAttribute(mxfp4_v4_grouped_rows_pack_rowcol_kernel<QuantMode::ENCODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaFuncSetAttribute(mxfp4_v4_grouped_rows_pack_rowcol_kernel<QuantMode::DECODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    dim3 grid(output_cols / CHUNK_DIM, total_padded / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_grouped_rows_pack_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                tma_in,
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
        case 2:
            mxfp4_v4_grouped_rows_pack_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                tma_in,
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
        default:
            mxfp4_v4_grouped_rows_pack_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                tma_in,
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 grouped pack+rowcol quant: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_pack_indexed_scaled_rows_quantize_row_and_col(
    torch::Tensor input,
    torch::Tensor token_indices,
    torch::Tensor scores,
    int64_t num_batches,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t output_cols,
    int mode
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(token_indices.is_cuda() && token_indices.is_contiguous());
    TORCH_CHECK(scores.is_cuda() && scores.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt64 && token_indices.dim() == 1);
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.dim() == 1);
    TORCH_CHECK(num_batches > 0, "num_batches must be positive");
    TORCH_CHECK(live_rows_per_batch > 0, "live_rows_per_batch must be positive");
    TORCH_CHECK(padded_rows_per_batch >= live_rows_per_batch,
                "padded rows must cover live rows");
    TORCH_CHECK(token_indices.size(0) >= num_batches * live_rows_per_batch,
                "token_indices does not cover the requested live rows");
    TORCH_CHECK(scores.size(0) >= num_batches * live_rows_per_batch,
                "scores does not cover the requested live rows");
    TORCH_CHECK(output_cols >= input.size(1), "output_cols must be at least input cols");
    TORCH_CHECK(input.size(1) % 128 == 0, "input cols must be 128-aligned");
    TORCH_CHECK(output_cols % 128 == 0 && padded_rows_per_batch % 128 == 0,
                "indexed scaled grouped quant requires 128-aligned output dims");
    auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(set_device_err));

    const int64_t total_padded = num_batches * padded_rows_per_batch;
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto row_fp4 = torch::empty(
        {total_padded, output_cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {total_padded / 128, output_cols / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {output_cols, total_padded / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {output_cols / 128, total_padded / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), total_padded, output_cols,
                  TILE_DIM, TILE_DIM, output_cols, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), output_cols, total_padded,
                  TILE_DIM, TILE_DIM, total_padded, 4);

    const int dshmem = v4_rowcol_shmem_size();
    cudaFuncSetAttribute(mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel<QuantMode::RTE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaFuncSetAttribute(mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel<QuantMode::ENCODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaFuncSetAttribute(mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel<QuantMode::DECODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    dim3 grid(output_cols / CHUNK_DIM, total_padded / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                token_indices.data_ptr<int64_t>(),
                scores.data_ptr<float>(),
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
        case 2:
            mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                token_indices.data_ptr<int64_t>(),
                scores.data_ptr<float>(),
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
        default:
            mxfp4_v4_indexed_scaled_rows_pack_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                token_indices.data_ptr<int64_t>(),
                scores.data_ptr<float>(),
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 indexed scaled pack+rowcol quant: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor inv_rms,
    torch::Tensor token_indices,
    int64_t num_batches,
    int64_t live_rows_per_batch,
    int64_t padded_rows_per_batch,
    int64_t output_cols,
    int mode
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous());
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.is_contiguous());
    TORCH_CHECK(token_indices.is_cuda() && token_indices.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16 && norm_weight.dim() == 1);
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32 && inv_rms.dim() == 1);
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt64 && token_indices.dim() == 1);
    TORCH_CHECK(norm_weight.size(0) == input.size(1), "norm_weight must match input cols");
    TORCH_CHECK(inv_rms.size(0) == input.size(0), "inv_rms must match input rows");
    TORCH_CHECK(num_batches > 0, "num_batches must be positive");
    TORCH_CHECK(live_rows_per_batch > 0, "live_rows_per_batch must be positive");
    TORCH_CHECK(padded_rows_per_batch >= live_rows_per_batch,
                "padded rows must cover live rows");
    TORCH_CHECK(token_indices.size(0) >= num_batches * live_rows_per_batch,
                "token_indices does not cover the requested live rows");
    TORCH_CHECK(output_cols >= input.size(1), "output_cols must be at least input cols");
    TORCH_CHECK(input.size(1) % 128 == 0, "input cols must be 128-aligned");
    TORCH_CHECK(output_cols % 128 == 0 && padded_rows_per_batch % 128 == 0,
                "indexed rmsnorm grouped quant requires 128-aligned output dims");
    auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(set_device_err));

    const int64_t total_padded = num_batches * padded_rows_per_batch;
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto row_fp4 = torch::empty(
        {total_padded, output_cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {total_padded / 128, output_cols / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {output_cols, total_padded / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {output_cols / 128, total_padded / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), total_padded, output_cols,
                  TILE_DIM, TILE_DIM, output_cols, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), output_cols, total_padded,
                  TILE_DIM, TILE_DIM, total_padded, 4);

    const int dshmem = v4_rowcol_shmem_size();
    cudaFuncSetAttribute(mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel<QuantMode::RTE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaFuncSetAttribute(mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel<QuantMode::ENCODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    cudaFuncSetAttribute(mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel<QuantMode::DECODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    dim3 grid(output_cols / CHUNK_DIM, total_padded / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                reinterpret_cast<const IType*>(norm_weight.data_ptr()),
                inv_rms.data_ptr<float>(),
                token_indices.data_ptr<int64_t>(),
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
        case 2:
            mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                reinterpret_cast<const IType*>(norm_weight.data_ptr()),
                inv_rms.data_ptr<float>(),
                token_indices.data_ptr<int64_t>(),
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
        default:
            mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(input.data_ptr()),
                reinterpret_cast<const IType*>(norm_weight.data_ptr()),
                inv_rms.data_ptr<float>(),
                token_indices.data_ptr<int64_t>(),
                tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                num_batches, live_rows_per_batch, padded_rows_per_batch,
                input.size(1), output_cols);
            break;
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 indexed rmsnorm pack+rowcol quant: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable(
    torch::Tensor input,
    torch::Tensor token_indices,
    torch::Tensor scores,
    std::vector<int64_t> route_starts,
    std::vector<int64_t> rows,
    std::vector<int64_t> padded_starts,
    std::vector<int64_t> padded_rows,
    int64_t output_cols,
    int mode
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(token_indices.is_cuda() && token_indices.is_contiguous());
    TORCH_CHECK(scores.is_cuda() && scores.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt64 && token_indices.dim() == 1);
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.dim() == 1);
    const size_t n = rows.size();
    TORCH_CHECK(n > 0, "expected at least one active group");
    TORCH_CHECK(route_starts.size() == n && padded_starts.size() == n && padded_rows.size() == n,
                "route_starts, rows, padded_starts, and padded_rows length mismatch");
    TORCH_CHECK(output_cols >= input.size(1), "output_cols must be at least input cols");
    TORCH_CHECK(input.size(1) % 128 == 0, "input cols must be 128-aligned");
    TORCH_CHECK(output_cols % 128 == 0, "output_cols must be 128-aligned");

    int64_t total_padded = 0;
    int64_t max_padded = 0;
    int64_t total_live = 0;
    for (size_t i = 0; i < n; ++i) {
        TORCH_CHECK(rows[i] >= 0 && padded_rows[i] >= rows[i], "invalid variable grouped row counts");
        TORCH_CHECK(padded_rows[i] % 128 == 0 && padded_starts[i] % 128 == 0,
                    "variable indexed grouped quant requires 128-aligned padded rows/starts");
        TORCH_CHECK(route_starts[i] >= 0, "route starts must be non-negative");
        total_live = std::max(total_live, route_starts[i] + rows[i]);
        total_padded = std::max(total_padded, padded_starts[i] + padded_rows[i]);
        max_padded = std::max(max_padded, padded_rows[i]);
    }
    TORCH_CHECK(total_padded > 0 && total_padded % 128 == 0, "total padded rows must be positive and 128-aligned");
    TORCH_CHECK(token_indices.size(0) >= total_live, "token_indices does not cover requested variable rows");
    TORCH_CHECK(scores.size(0) >= total_live, "scores does not cover requested variable rows");
    auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(set_device_err));

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto row_fp4 = torch::empty(
        {total_padded, output_cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {total_padded / 128, output_cols / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {output_cols, total_padded / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {output_cols / 128, total_padded / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    auto route_starts_dev = make_i64_device_tensor(route_starts, device, stream);
    auto rows_dev = make_i64_device_tensor(rows, device, stream);
    auto padded_starts_dev = make_i64_device_tensor(padded_starts, device, stream);
    auto padded_rows_dev = make_i64_device_tensor(padded_rows, device, stream);

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), total_padded, output_cols,
                  TILE_DIM, TILE_DIM, output_cols, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), output_cols, total_padded,
                  TILE_DIM, TILE_DIM, total_padded, 4);

    TORCH_CHECK(mode == 1, "variable indexed scaled grouped quant currently supports encode mode only");
    const int dshmem = v4_rowcol_shmem_size();
    cudaFuncSetAttribute(mxfp4_v4_indexed_scaled_rows_pack_rowcol_variable_kernel<QuantMode::ENCODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    dim3 grid(output_cols / CHUNK_DIM, max_padded / CHUNK_DIM, static_cast<unsigned int>(n));
    mxfp4_v4_indexed_scaled_rows_pack_rowcol_variable_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
        reinterpret_cast<const IType*>(input.data_ptr()),
        token_indices.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        route_starts_dev.data_ptr<int64_t>(),
        rows_dev.data_ptr<int64_t>(),
        padded_starts_dev.data_ptr<int64_t>(),
        padded_rows_dev.data_ptr<int64_t>(),
        tma_row_out, tma_col_out,
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        total_padded, input.size(1), output_cols);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 variable indexed scaled pack+rowcol quant: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor inv_rms,
    torch::Tensor token_indices,
    std::vector<int64_t> route_starts,
    std::vector<int64_t> rows,
    std::vector<int64_t> padded_starts,
    std::vector<int64_t> padded_rows,
    int64_t output_cols,
    int mode
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous());
    TORCH_CHECK(inv_rms.is_cuda() && inv_rms.is_contiguous());
    TORCH_CHECK(token_indices.is_cuda() && token_indices.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16 && norm_weight.dim() == 1);
    TORCH_CHECK(inv_rms.scalar_type() == torch::kFloat32 && inv_rms.dim() == 1);
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt64 && token_indices.dim() == 1);
    TORCH_CHECK(norm_weight.size(0) == input.size(1), "norm_weight must match input cols");
    TORCH_CHECK(inv_rms.size(0) == input.size(0), "inv_rms must match input rows");
    const size_t n = rows.size();
    TORCH_CHECK(n > 0, "expected at least one active group");
    TORCH_CHECK(route_starts.size() == n && padded_starts.size() == n && padded_rows.size() == n,
                "route_starts, rows, padded_starts, and padded_rows length mismatch");
    TORCH_CHECK(output_cols >= input.size(1), "output_cols must be at least input cols");
    TORCH_CHECK(input.size(1) % 128 == 0, "input cols must be 128-aligned");
    TORCH_CHECK(output_cols % 128 == 0, "output_cols must be 128-aligned");

    int64_t total_padded = 0;
    int64_t max_padded = 0;
    int64_t total_live = 0;
    for (size_t i = 0; i < n; ++i) {
        TORCH_CHECK(rows[i] >= 0 && padded_rows[i] >= rows[i], "invalid variable grouped row counts");
        TORCH_CHECK(padded_rows[i] % 128 == 0 && padded_starts[i] % 128 == 0,
                    "variable indexed grouped quant requires 128-aligned padded rows/starts");
        TORCH_CHECK(route_starts[i] >= 0, "route starts must be non-negative");
        total_live = std::max(total_live, route_starts[i] + rows[i]);
        total_padded = std::max(total_padded, padded_starts[i] + padded_rows[i]);
        max_padded = std::max(max_padded, padded_rows[i]);
    }
    TORCH_CHECK(total_padded > 0 && total_padded % 128 == 0, "total padded rows must be positive and 128-aligned");
    TORCH_CHECK(token_indices.size(0) >= total_live, "token_indices does not cover requested variable rows");
    auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed: ", cudaGetErrorString(set_device_err));

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto row_fp4 = torch::empty(
        {total_padded, output_cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {total_padded / 128, output_cols / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {output_cols, total_padded / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {output_cols / 128, total_padded / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    auto route_starts_dev = make_i64_device_tensor(route_starts, device, stream);
    auto rows_dev = make_i64_device_tensor(rows, device, stream);
    auto padded_starts_dev = make_i64_device_tensor(padded_starts, device, stream);
    auto padded_rows_dev = make_i64_device_tensor(padded_rows, device, stream);

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), total_padded, output_cols,
                  TILE_DIM, TILE_DIM, output_cols, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), output_cols, total_padded,
                  TILE_DIM, TILE_DIM, total_padded, 4);

    TORCH_CHECK(mode == 1, "variable indexed rmsnorm grouped quant currently supports encode mode only");
    const int dshmem = v4_rowcol_shmem_size();
    cudaFuncSetAttribute(mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_variable_kernel<QuantMode::ENCODE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);

    dim3 grid(output_cols / CHUNK_DIM, max_padded / CHUNK_DIM, static_cast<unsigned int>(n));
    mxfp4_v4_indexed_rmsnorm_rows_pack_rowcol_variable_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
        reinterpret_cast<const IType*>(input.data_ptr()),
        reinterpret_cast<const IType*>(norm_weight.data_ptr()),
        inv_rms.data_ptr<float>(),
        token_indices.data_ptr<int64_t>(),
        route_starts_dev.data_ptr<int64_t>(),
        rows_dev.data_ptr<int64_t>(),
        padded_starts_dev.data_ptr<int64_t>(),
        padded_rows_dev.data_ptr<int64_t>(),
        tma_row_out, tma_col_out,
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        total_padded, input.size(1), output_cols);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 variable indexed rmsnorm pack+rowcol quant: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

void mxfp4_scatter_grouped_rows_bf16(
    torch::Tensor input,
    torch::Tensor output,
    std::vector<int64_t> starts,
    std::vector<int64_t> rows,
    std::vector<int64_t> padded_rows
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(output.is_cuda() && output.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16 && output.dim() == 2);
    TORCH_CHECK(starts.size() == rows.size() && rows.size() == padded_rows.size(),
                "starts, rows, and padded_rows length mismatch");
    const int64_t n = static_cast<int64_t>(rows.size());
    TORCH_CHECK(n > 0, "expected at least one group");
    const int64_t input_cols = input.size(1);
    const int64_t output_cols = output.size(1);
    int64_t total_padded = 0;
    int64_t max_live = 0;
    std::vector<int64_t> padded_starts;
    padded_starts.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        TORCH_CHECK(rows[i] >= 0 && padded_rows[i] >= rows[i], "invalid grouped row counts");
        TORCH_CHECK(starts[i] >= 0 && starts[i] + rows[i] <= output.size(0), "group row range out of output bounds");
        padded_starts.push_back(total_padded);
        total_padded += padded_rows[i];
        max_live = std::max(max_live, rows[i]);
    }
    TORCH_CHECK(total_padded <= input.size(0), "packed input does not cover requested groups");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto starts_dev = make_i64_device_tensor(starts, input.device(), stream);
    auto rows_dev = make_i64_device_tensor(rows, input.device(), stream);
    auto padded_starts_dev = make_i64_device_tensor(padded_starts, input.device(), stream);
    constexpr int threads = 256;
    const bool use_vec = (input_cols % 8 == 0) && (output_cols % 8 == 0);
    const int64_t input_cols_work = use_vec ? input_cols / 8 : input_cols;
    const int64_t output_cols_work = use_vec ? output_cols / 8 : output_cols;
    const int64_t work_per_group = std::max<int64_t>(1, max_live * output_cols_work);
    const int grid_y = static_cast<int>(std::min<int64_t>(1024, (work_per_group + threads - 1) / threads));
    dim3 grid(n, grid_y);
    if (use_vec) {
        mxfp4_scatter_grouped_rows_bf16_vec_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const int4*>(input.data_ptr()),
            reinterpret_cast<int4*>(output.data_ptr()),
            starts_dev.data_ptr<int64_t>(),
            rows_dev.data_ptr<int64_t>(),
            padded_starts_dev.data_ptr<int64_t>(),
            input_cols_work,
            output_cols_work,
            max_live);
    } else {
        mxfp4_scatter_grouped_rows_bf16_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const IType*>(input.data_ptr()),
            reinterpret_cast<IType*>(output.data_ptr()),
            starts_dev.data_ptr<int64_t>(),
            rows_dev.data_ptr<int64_t>(),
            padded_starts_dev.data_ptr<int64_t>(),
            input_cols,
            output_cols,
            max_live);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 grouped bf16 row scatter: ", cudaGetErrorString(err));
}

__global__ void mxfp4_copy_col_fp4_slices_kernel(
    const uint8_t* __restrict__ src,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ row_starts,
    const int64_t* __restrict__ rows,
    int64_t K,
    int64_t src_packed_cols
) {
    const int expert = blockIdx.y;
    const int64_t packed_cols = rows[expert] / 2;
    const int64_t packed_start = row_starts[expert] / 2;
    const int64_t total = K * packed_cols;
    uint8_t* dst = reinterpret_cast<uint8_t*>(dst_ptrs[expert]);
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int64_t k = linear / packed_cols;
        const int64_t c = linear - k * packed_cols;
        dst[linear] = src[k * src_packed_cols + packed_start + c];
    }
}

__global__ void mxfp4_copy_col_scale_slices_kernel(
    const uint8_t* __restrict__ src,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ row_starts,
    const int64_t* __restrict__ rows,
    int64_t K_blocks,
    int64_t src_row_blocks
) {
    const int expert = blockIdx.y;
    const int64_t row_blocks = rows[expert] / 128;
    const int64_t row_block_start = row_starts[expert] / 128;
    constexpr int64_t SCALE_INNER = 32 * 16;
    const int64_t total = K_blocks * row_blocks * SCALE_INNER;
    uint8_t* dst = reinterpret_cast<uint8_t*>(dst_ptrs[expert]);
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        const int64_t inner = linear % SCALE_INNER;
        const int64_t tmp = linear / SCALE_INNER;
        const int64_t rb = tmp % row_blocks;
        const int64_t kb = tmp / row_blocks;
        dst[linear] = src[(kb * src_row_blocks + row_block_start + rb) * SCALE_INNER + inner];
    }
}

std::vector<std::vector<torch::Tensor>> mxfp4_copy_col_slices(
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    std::vector<int64_t> row_starts,
    std::vector<int64_t> rows
) {
    TORCH_CHECK(col_fp4.is_cuda(), "col_fp4 must be CUDA");
    TORCH_CHECK(col_sc.is_cuda(), "col_sc must be CUDA");
    TORCH_CHECK(col_fp4.is_contiguous(), "col_fp4 must be contiguous");
    TORCH_CHECK(col_sc.is_contiguous(), "col_sc must be contiguous");
    TORCH_CHECK(row_starts.size() == rows.size(), "row_starts and rows length mismatch");
    const int64_t n = static_cast<int64_t>(rows.size());
    TORCH_CHECK(n > 0, "expected at least one slice");
    TORCH_CHECK(col_fp4.dim() == 2, "col_fp4 must be [K, M/2]");
    TORCH_CHECK(col_sc.dim() == 4, "col_sc must be [K/128, M/128, 32, 16]");
    TORCH_CHECK(col_sc.size(2) == 32 && col_sc.size(3) == 16, "col_sc inner shape must be 32x16");

    const int64_t K = col_fp4.size(0);
    const int64_t src_packed_cols = col_fp4.size(1);
    const int64_t K_blocks = col_sc.size(0);
    const int64_t src_row_blocks = col_sc.size(1);
    std::vector<torch::Tensor> fp4_out;
    std::vector<torch::Tensor> sc_out;
    fp4_out.reserve(n);
    sc_out.reserve(n);

    std::vector<int64_t> fp4_ptrs(n);
    std::vector<int64_t> sc_ptrs(n);
    for (int64_t i = 0; i < n; ++i) {
        TORCH_CHECK(row_starts[i] % 128 == 0, "row_start must be 128-aligned");
        TORCH_CHECK(rows[i] > 0 && rows[i] % 128 == 0, "rows must be positive and 128-aligned");
        TORCH_CHECK(row_starts[i] + rows[i] <= src_packed_cols * 2, "slice exceeds col_fp4 width");
        auto fp4 = torch::empty({K, rows[i] / 2}, col_fp4.options());
        auto sc = torch::empty({K_blocks, rows[i] / 128, 32, 16}, col_sc.options());
        fp4_ptrs[i] = reinterpret_cast<int64_t>(fp4.data_ptr());
        sc_ptrs[i] = reinterpret_cast<int64_t>(sc.data_ptr());
        fp4_out.push_back(fp4);
        sc_out.push_back(sc);
    }

    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(col_fp4.device());
    auto fp4_ptrs_dev = torch::empty({n}, long_opts);
    auto sc_ptrs_dev = torch::empty({n}, long_opts);
    auto starts_dev = torch::empty({n}, long_opts);
    auto rows_dev = torch::empty({n}, long_opts);
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(fp4_ptrs_dev.data_ptr<int64_t>(), fp4_ptrs.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream());
    cudaMemcpyAsync(sc_ptrs_dev.data_ptr<int64_t>(), sc_ptrs.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream());
    cudaMemcpyAsync(starts_dev.data_ptr<int64_t>(), row_starts.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream());
    cudaMemcpyAsync(rows_dev.data_ptr<int64_t>(), rows.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream());

    constexpr int threads = 256;
    int64_t max_fp4 = 0;
    int64_t max_sc = 0;
    for (int64_t rows_i : rows) {
        max_fp4 = std::max(max_fp4, K * (rows_i / 2));
        max_sc = std::max(max_sc, K_blocks * (rows_i / 128) * 32 * 16);
    }
    dim3 fp4_grid(static_cast<unsigned>(std::min<int64_t>((max_fp4 + threads - 1) / threads, 65535)), static_cast<unsigned>(n));
    dim3 sc_grid(static_cast<unsigned>(std::min<int64_t>((max_sc + threads - 1) / threads, 65535)), static_cast<unsigned>(n));
    mxfp4_copy_col_fp4_slices_kernel<<<fp4_grid, threads, 0, stream.stream()>>>(
        reinterpret_cast<const uint8_t*>(col_fp4.data_ptr()),
        fp4_ptrs_dev.data_ptr<int64_t>(),
        starts_dev.data_ptr<int64_t>(),
        rows_dev.data_ptr<int64_t>(),
        K,
        src_packed_cols
    );
    mxfp4_copy_col_scale_slices_kernel<<<sc_grid, threads, 0, stream.stream()>>>(
        reinterpret_cast<const uint8_t*>(col_sc.data_ptr()),
        sc_ptrs_dev.data_ptr<int64_t>(),
        starts_dev.data_ptr<int64_t>(),
        rows_dev.data_ptr<int64_t>(),
        K_blocks,
        src_row_blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fp4_out, sc_out};
}

template <int BLOCK_SIZE = 256>
__global__ void mxfp4_fused_rmsnorm_to_bf16_kernel(
    const IType* __restrict__ x,
    const IType* __restrict__ gamma,
    IType* __restrict__ out,
    float* __restrict__ inv_rms_out,
    float epsilon,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const IType* row_x = x + static_cast<int64_t>(row) * cols;
    IType* row_out = out + static_cast<int64_t>(row) * cols;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        const float v = __bfloat162float(row_x[i]);
        sum_sq += v * v;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    }

    __shared__ float warp_sums[BLOCK_SIZE / 32];
    __shared__ float row_inv_rms;
    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_sums[wid] = sum_sq;
    }
    __syncthreads();

    if (wid == 0) {
        sum_sq = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int mask = (BLOCK_SIZE / 32) / 2; mask > 0; mask >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        }
        if (lane == 0) {
            row_inv_rms = rsqrtf(sum_sq / cols + epsilon);
            inv_rms_out[row] = row_inv_rms;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        const float v = __bfloat162float(row_x[i]);
        const float g = __bfloat162float(gamma[i]);
        row_out[i] = __float2bfloat16_rn(v * row_inv_rms * g);
    }
}

// ═══════════════════════════════════════════════════════════════════
// TMA helper
// ═══════════════════════════════════════════════════════════════════
static void create_tma_2d(
    CUtensorMap& tma,
    void* ptr,
    uint64_t dimY, uint64_t dimX,
    uint32_t boxY, uint32_t boxX,
    uint64_t strideX, size_t elemBits,
    CUtensorMapL2promotion l2promo
) {
    typedef CUresult (*cuTensorMapEncodeTiled_t)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
        const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
        const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (!fn) {
        void *handle = dlopen("libcuda.so.1", RTLD_LAZY);
        TORCH_CHECK(handle, "Failed to open libcuda.so.1");
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(handle, "cuTensorMapEncodeTiled"));
        TORCH_CHECK(fn, "cuTensorMapEncodeTiled not found");
    }

    CUtensorMapDataType dtype;
    if (elemBits == 16) dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if (elemBits == 8) dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    else if (elemBits == 4) dtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    else TORCH_CHECK(false, "Unsupported elem bits: ", elemBits);

    uint64_t size[2]   = {dimX, dimY};
    uint64_t stride[1] = {strideX * elemBits / 8};
    uint32_t box[2]    = {boxX, boxY};
    uint32_t elStride[2] = {1, 1};
    auto result = fn(&tma, dtype, 2, ptr,
        size, stride, box, elStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
}

static bool is_power_of_two_int64(int64_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static void check_rope_live64_tensor(
    const torch::Tensor& rope_cs,
    int64_t rope_seq_len
) {
    TORCH_CHECK(rope_cs.is_cuda() && rope_cs.is_contiguous());
    TORCH_CHECK(rope_cs.scalar_type() == torch::kFloat32);
    TORCH_CHECK(rope_cs.dim() == 3);
    TORCH_CHECK(is_power_of_two_int64(rope_seq_len),
                "rope_seq_len must be a positive power of two");
    TORCH_CHECK(
        rope_cs.sizes() == torch::IntArrayRef({rope_seq_len, 32, 2}),
        "rope_cs must have shape (rope_seq_len, 32, 2)");
}

// ═══════════════════════════════════════════════════════════════════
// Cached SM count + persistent threshold
// ═══════════════════════════════════════════════════════════════════
struct CachedInfo {
    int num_sms = 0;
    int max_bps = 0;   // max blocks per SM (fused kernel)
    int p_max_bps = 0; // max blocks per SM (persistent kernel)
    int dshmem = 0;
    bool initialized = false;
};

static constexpr int MAX_CUDA_DEVICES = 16;

struct DeviceInitFlags {
    std::array<bool, MAX_CUDA_DEVICES> initialized{};
};

template <typename InitFn>
static void ensure_device_init(DeviceInitFlags& flags, InitFn&& init_fn) {
    int dev = 0;
    auto err = cudaGetDevice(&dev);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    TORCH_CHECK(dev >= 0 && dev < MAX_CUDA_DEVICES,
                "unsupported CUDA device index for cached kernel init: ", dev);
    if (!flags.initialized[dev]) {
        init_fn();
        flags.initialized[dev] = true;
    }
}

static CachedInfo& get_cached() {
    static std::array<CachedInfo, MAX_CUDA_DEVICES> cache;
    int dev = 0;
    auto err = cudaGetDevice(&dev);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    TORCH_CHECK(dev >= 0 && dev < MAX_CUDA_DEVICES,
                "unsupported CUDA device index for cached occupancy: ", dev);
    auto& ci = cache[dev];
    if (!ci.initialized) {
        cudaDeviceGetAttribute(&ci.num_sms, cudaDevAttrMultiProcessorCount, dev);
        ci.dshmem = v3_shmem_size();

        // Fused kernel occupancy (default RTE template)
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ci.max_bps, mxfp4_v3_fused_kernel<QuantMode::RTE>, THREADS, ci.dshmem);

        // Set attributes for other mode instantiations too
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);

        // Persistent kernel occupancy
        cudaFuncSetAttribute(mxfp4_v3_persistent_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ci.dshmem);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ci.p_max_bps, mxfp4_v3_persistent_kernel<QuantMode::RTE>, THREADS, ci.dshmem);

        ci.initialized = true;
    }
    return ci;
}

// NOTE: Persistent kernel disabled pending non-determinism fix.
// Fused kernel has same pipelining and is bit-exact with v2.
static constexpr int PERSISTENT_THRESHOLD = 999999999;  // effectively disabled

// ═══════════════════════════════════════════════════════════════════
// Global work counter for persistent kernel
// ═══════════════════════════════════════════════════════════════════
static unsigned int* g_work_counter = nullptr;
static void ensure_work_counter() {
    if (!g_work_counter) {
        cudaMalloc(&g_work_counter, sizeof(unsigned int));
    }
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void mxfp4_load_nhsd_wo_tile(
    InputBuf3D& sIn,
    uint64_t* in_mbar,
    const CUtensorMap& tensor_map_input,
    int logical_block_offset_Y,
    int logical_block_offset_X,
    int tile_id,
    int H,
    int S,
    int D
) {
    (void)MODE;
    const int stage_Y = tile_id / TILES_X;
    const int stage_X = tile_id % TILES_X;
    const int logical_row = logical_block_offset_Y + stage_Y * TILE_DIM;
    const int logical_col = logical_block_offset_X + stage_X * TILE_DIM;
    const int b = logical_row / S;
    const int s = logical_row - b * S;
    const int h = logical_col / D;
    const int d = logical_col - h * D;
    const int input_row = (b * H + h) * S + s;

    mbarrier_arrive_expect_tx(&in_mbar[tile_id], TILE_DIM * TILE_DIM * sizeof(IType));
    cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t*>(&sIn[tile_id]),
        reinterpret_cast<const uint64_t*>(&tensor_map_input),
        d,
        input_row,
        &in_mbar[tile_id]);
}

template<QuantMode MODE = QuantMode::RTE>
__device__ __forceinline__ void mxfp4_quantize_nhsd_wo_rowcol_pipelined(
    IType* sIn_ptr,
    fp4e2m1x2* sOut_ptr,
    uint8_t* row_scale_buf,
    uint8_t* col_scale_buf,
    OutputBuf3D& sOut,
    const CUtensorMap& tensor_map_input,
    const CUtensorMap& tensor_map_row_output,
    const CUtensorMap& tensor_map_col_output,
    int block_offset_Y,
    int block_offset_X,
    int H,
    int S,
    int D,
    uint64_t* in_mbar,
    int mbar_phase
) {
    const bool leading = (threadIdx.x == 0);
    auto& sIn = *reinterpret_cast<InputBuf3D*>(sIn_ptr);
    constexpr int row_buff_out = 0;
    constexpr int col_buff_out = 1;

    #pragma unroll
    for (int pre = 0; pre < 2; ++pre) {
        if (leading) {
            mxfp4_load_nhsd_wo_tile<MODE>(
                sIn, in_mbar, tensor_map_input,
                block_offset_Y, block_offset_X,
                pre, H, S, D);
        }
    }

    #pragma unroll
    for (int t = 0; t < NUM_TILES; ++t) {
        const int stage_Y = t / TILES_X;
        const int stage_X = t % TILES_X;
        const int row_stage_offset_Y = stage_Y * TILE_DIM;
        const int row_stage_offset_X = stage_X * TILE_DIM;
        const int col_stage_offset_Y = stage_X * TILE_DIM;
        const int col_stage_offset_X = stage_Y * TILE_DIM;

        if (t + 2 < NUM_TILES && leading) {
            mxfp4_load_nhsd_wo_tile<MODE>(
                sIn, in_mbar, tensor_map_input,
                block_offset_Y, block_offset_X,
                t + 2, H, S, D);
        }

        mbarrier_wait_parity_acquire_cta_shared_cta(&in_mbar[t], mbar_phase);

        mx_rowwise_quantize<MODE>(
            sIn_ptr, sOut_ptr, row_scale_buf, stage_Y, stage_X, t, row_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_row_output),
                block_offset_X + row_stage_offset_X,
                block_offset_Y + row_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[row_buff_out]));
            cp_async_bulk_commit_group();
        }

        mx_colwise_quantize_direct<MODE>(
            sIn_ptr, sOut_ptr, col_scale_buf, stage_X, stage_Y, t, col_buff_out);

        fence_proxy_async_shared_cta();
        __syncthreads();

        if (leading) {
            cp_async_bulk_tensor_2d_shared_to_global(
                reinterpret_cast<const uint64_t*>(&tensor_map_col_output),
                block_offset_Y + col_stage_offset_X,
                block_offset_X + col_stage_offset_Y,
                reinterpret_cast<uint64_t*>(&sOut[col_buff_out]));
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_group_read<0>();
        }
        __syncthreads();
    }
}

template<QuantMode MODE = QuantMode::RTE>
__global__ void __launch_bounds__(THREADS)
mxfp4_v4_nhsd_wo_rowcol_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_row_output,
    const __grid_constant__ CUtensorMap tensor_map_col_output,
    uint8_t* __restrict__ row_scales_out,
    uint8_t* __restrict__ col_scales_out,
    const int64_t M,
    const int64_t K,
    const int B,
    const int H,
    const int S,
    const int D
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    (void)B;
    const bool leading = (threadIdx.x == 0);
    const int row_ntk = K / CHUNK_DIM;
    const int col_ntk = M / CHUNK_DIM;
    const int ctaid_X = blockIdx.x;
    const int ctaid_Y = blockIdx.y;
    const int block_offset_Y = ctaid_Y * CHUNK_DIM;
    const int block_offset_X = ctaid_X * CHUNK_DIM;

    constexpr int in_bytes  = DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT);
    constexpr int out_bytes = DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT);
    constexpr int sc_bytes  = DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    extern __shared__ unsigned char dynamic_shmem[];
    unsigned char* dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

    IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
    fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + in_bytes);
    uint8_t* row_scale_buf = dshmem + in_bytes + out_bytes;
    uint8_t* col_scale_buf = row_scale_buf + sc_bytes;

    auto& sOut = *reinterpret_cast<OutputBuf3D*>(sOut_ptr);

    __shared__ uint64_t in_mbar[NUM_TILES];
    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_init(&in_mbar[t], 1);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    mxfp4_quantize_nhsd_wo_rowcol_pipelined<MODE>(
        sIn_ptr,
        sOut_ptr,
        row_scale_buf,
        col_scale_buf,
        sOut,
        tensor_map_input,
        tensor_map_row_output,
        tensor_map_col_output,
        block_offset_Y,
        block_offset_X,
        H,
        S,
        D,
        in_mbar,
        0);

    write_scales_swizzled(row_scale_buf, row_scales_out, ctaid_X, ctaid_Y, row_ntk);
    write_scales_swizzled(col_scale_buf, col_scales_out, ctaid_Y, ctaid_X, col_ntk);

    if (leading) {
        #pragma unroll
        for (int t = 0; t < NUM_TILES; ++t) {
            mbarrier_invalid(&in_mbar[t]);
        }
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════
// Single tensor quantize
// ═══════════════════════════════════════════════════════════════════
// Templated quantize function
template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_impl(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;
    const int total_tiles = tiles_X * tiles_Y;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    // TMA maps: use TILE_DIM (64) as box size for sub-tile pipelining
    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 16,
                  total_tiles >= PERSISTENT_THRESHOLD ?
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B :
                      CU_TENSOR_MAP_L2_PROMOTION_NONE);
    create_tma_2d(tma_out, fp4_out.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 4);

    if (total_tiles >= PERSISTENT_THRESHOLD) {
        // Persistent path
        ensure_work_counter();
        cudaMemsetAsync(g_work_counter, 0, sizeof(unsigned int), stream);

        int num_persistent = ci.p_max_bps * ci.num_sms;
        if (num_persistent > total_tiles) num_persistent = total_tiles;

        PersistentArgs args;
        args.work_counter = g_work_counter;
        args.tiles_X = tiles_X;
        args.tiles_Y = tiles_Y;
        args.total_tiles = total_tiles;

        mxfp4_v3_persistent_kernel<MODE><<<num_persistent, THREADS, dshmem, stream>>>(
            tma_in, tma_out,
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K, args);
    } else {
        // Fused path — one CTA per chunk
        dim3 grid(tiles_X, tiles_Y);
        mxfp4_v3_fused_kernel<MODE><<<grid, THREADS, dshmem, stream>>>(
            tma_in, tma_out,
            reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 quantize: ", cudaGetErrorString(err));

    return std::make_tuple(fp4_out, sc_out);
}

// Python-facing dispatch by mode int
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm(torch::Tensor input, int mode) {
    switch (mode) {
        case 1: return mxfp4_quantize_for_gemm_impl<QuantMode::ENCODE>(input);
        case 2: return mxfp4_quantize_for_gemm_impl<QuantMode::DECODE>(input);
        default: return mxfp4_quantize_for_gemm_impl<QuantMode::RTE>(input);
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_opt_impl(
    torch::Tensor input,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before MXFP4 opt row quantization: ",
                cudaGetErrorString(set_device_err));
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    static DeviceInitFlags quant_opt_init;
    ensure_device_init(quant_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v3_fused_kernel_opt<MODE, DATA_SR, SCALE_SR>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_out.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);

    dim3 grid(tiles_X, tiles_Y);
    mxfp4_v3_fused_kernel_opt<MODE, DATA_SR, SCALE_SR><<<grid, THREADS, dshmem, stream>>>(
        tma_in,
        tma_out,
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        M,
        K,
        rng_seed,
        rng_subsequence
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 quantize opt: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_opt(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    if (!data_stochastic_rounding && !scale_stochastic_rounding) {
        return mxfp4_quantize_for_gemm(input, mode);
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::ENCODE, true, true>(input, rng_seed, rng_subsequence);
            case 2: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::DECODE, true, true>(input, rng_seed, rng_subsequence);
            default: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::RTE, true, true>(input, rng_seed, rng_subsequence);
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::ENCODE, true, false>(input, rng_seed, rng_subsequence);
            case 2: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::DECODE, true, false>(input, rng_seed, rng_subsequence);
            default: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::RTE, true, false>(input, rng_seed, rng_subsequence);
        }
    }
    switch (mode) {
        case 1: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::ENCODE, false, true>(input, rng_seed, rng_subsequence);
        case 2: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::DECODE, false, true>(input, rng_seed, rng_subsequence);
        default: return mxfp4_quantize_for_gemm_opt_impl<QuantMode::RTE, false, true>(input, rng_seed, rng_subsequence);
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, int RHT_BLOCK_SIZE, bool WITH_RANDOM_SIGN_MASK>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_opt_rht_impl(
    torch::Tensor input,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(K % RHT_BLOCK_SIZE == 0, "K must be divisible by the RHT block size");

    const c10::cuda::CUDAGuard device_guard(input.device());
    const auto set_device_err = cudaSetDevice(input.get_device());
    TORCH_CHECK(set_device_err == cudaSuccess,
                "cudaSetDevice failed before MXFP4 opt-RHT row quantization: ",
                cudaGetErrorString(set_device_err));
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    auto fp4_out = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    static DeviceInitFlags quant_rht_opt_init;
    ensure_device_init(quant_rht_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v3_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, true, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_out.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);

    dim3 grid(tiles_X, tiles_Y);
    mxfp4_v3_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, true, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK><<<grid, THREADS, dshmem, stream>>>(
        tma_in,
        tma_out,
        reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
        M,
        K,
        rng_seed,
        rng_subsequence
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 quantize fused-rht opt: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_for_gemm_opt_rht(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(rht_block_size == 16 || rht_block_size == 32, "RHT block size must be 16 or 32");
#define MXFP4_DISPATCH_RHT_CASE(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_for_gemm_opt_rht_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, RHT_BLOCK, SIGN_FLAG>(input, rng_seed, rng_subsequence)

    if (rht_block_size == 16) {
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, true, 16, true);
                    case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, true, 16, true);
                    default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, true, 16, true);
                }
            }
            if (data_stochastic_rounding) {
                switch (mode) {
                    case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, false, 16, true);
                    case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, false, 16, true);
                    default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, false, 16, true);
                }
            }
            if (scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, true, 16, true);
                    case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, true, 16, true);
                    default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, true, 16, true);
                }
            }
            switch (mode) {
                case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, false, 16, true);
                case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, false, 16, true);
                default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, false, 16, true);
            }
        } else {
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, true, 16, false);
                    case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, true, 16, false);
                    default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, true, 16, false);
                }
            }
            if (data_stochastic_rounding) {
                switch (mode) {
                    case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, false, 16, false);
                    case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, false, 16, false);
                    default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, false, 16, false);
                }
            }
            if (scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, true, 16, false);
                    case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, true, 16, false);
                    default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, true, 16, false);
                }
            }
            switch (mode) {
                case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, false, 16, false);
                case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, false, 16, false);
                default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, false, 16, false);
            }
        }
    }

    if (with_random_sign_mask) {
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            switch (mode) {
                case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, true, 32, true);
                case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, true, 32, true);
                default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, true, 32, true);
            }
        }
        if (data_stochastic_rounding) {
            switch (mode) {
                case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, false, 32, true);
                case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, false, 32, true);
                default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, false, 32, true);
            }
        }
        if (scale_stochastic_rounding) {
            switch (mode) {
                case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, true, 32, true);
                case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, true, 32, true);
                default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, true, 32, true);
            }
        }
        switch (mode) {
            case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, false, 32, true);
            case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, false, 32, true);
            default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, false, 32, true);
        }
    }

    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, true, 32, false);
            case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, true, 32, false);
            default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, true, 32, false);
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, true, false, 32, false);
            case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, true, false, 32, false);
            default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, true, false, 32, false);
        }
    }
    if (scale_stochastic_rounding) {
        switch (mode) {
            case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, true, 32, false);
            case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, true, 32, false);
            default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, true, 32, false);
        }
    }
    switch (mode) {
        case 1: return MXFP4_DISPATCH_RHT_CASE(QuantMode::ENCODE, false, false, 32, false);
        case 2: return MXFP4_DISPATCH_RHT_CASE(QuantMode::DECODE, false, false, 32, false);
        default: return MXFP4_DISPATCH_RHT_CASE(QuantMode::RTE, false, false, 32, false);
    }
#undef MXFP4_DISPATCH_RHT_CASE
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT, int RHT_BLOCK_SIZE, bool WITH_RANDOM_SIGN_MASK>
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_group_quantize_dim0_opt_impl(
    torch::Tensor input,
    std::vector<int64_t> group_sizes,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % 128 == 0);
    if constexpr (WITH_RHT) {
        TORCH_CHECK(K % RHT_BLOCK_SIZE == 0, "K must be divisible by the RHT block size");
    }

    int ng = static_cast<int>(group_sizes.size());
    TORCH_CHECK(ng >= 1 && ng <= MAX_GROUPS);

    int64_t total = 0;
    for (auto s : group_sizes) {
        TORCH_CHECK(s % 128 == 0);
        total += s;
    }
    TORCH_CHECK(total == M);

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fp4_all = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));

    std::vector<torch::Tensor> sc_allocs(ng);
    GroupArgs args;
    memset(&args, 0, sizeof(args));
    args.num_groups = ng;
    args.boundaries[0] = 0;

    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        args.boundaries[i + 1] = args.boundaries[i] + static_cast<int>(Mi);
        sc_allocs[i] = torch::empty({Mi / 128, K / 128, 32, 16},
            torch::dtype(torch::kUInt8).device(device));
        args.scale_ptrs[i] = reinterpret_cast<uint8_t*>(sc_allocs[i].data_ptr());
    }

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    static DeviceInitFlags grp_opt_init;
    ensure_device_init(grp_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v3_fused_group_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_all.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);

    dim3 grid(tiles_X, tiles_Y);
    mxfp4_v3_fused_group_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in, tma_out, M, K, args, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 grouped quantize opt: ", cudaGetErrorString(err));

    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    int64_t row_off = 0;
    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        auto fp4_slice = fp4_all.narrow(0, row_off, Mi).contiguous();
        results.push_back(std::make_tuple(fp4_slice, sc_allocs[i]));
        row_off += Mi;
    }
    return results;
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_group_quantize_dim0_opt(
    torch::Tensor input,
    std::vector<int64_t> group_sizes,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define MXFP4_DISPATCH_GROUP_CASE(MODE_ENUM, DATA_SR, SCALE_SR, WITH_RHT, BS, WITH_SIGN) \
    return mxfp4_group_quantize_dim0_opt_impl<MODE_ENUM, DATA_SR, SCALE_SR, WITH_RHT, BS, WITH_SIGN>( \
        input, group_sizes, rng_seed, rng_subsequence)

    if (use_rht) {
        if (rht_block_size == 16) {
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                if (with_random_sign_mask) {
                    MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, true, true, 16, true);
                }
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, true, true, 16, false);
            }
            if (data_stochastic_rounding) {
                if (with_random_sign_mask) {
                    MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, false, true, 16, true);
                }
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, false, true, 16, false);
            }
            if (scale_stochastic_rounding) {
                if (with_random_sign_mask) {
                    MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, true, true, 16, true);
                }
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, true, true, 16, false);
            }
            if (with_random_sign_mask) {
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, false, true, 16, true);
            }
            MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, false, true, 16, false);
        }
        TORCH_CHECK(rht_block_size == 32, "RHT block size must be 16 or 32");
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            if (with_random_sign_mask) {
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, true, true, 32, true);
            }
            MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, true, true, 32, false);
        }
        if (data_stochastic_rounding) {
            if (with_random_sign_mask) {
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, false, true, 32, true);
            }
            MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, false, true, 32, false);
        }
        if (scale_stochastic_rounding) {
            if (with_random_sign_mask) {
                MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, true, true, 32, true);
            }
            MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, true, true, 32, false);
        }
        if (with_random_sign_mask) {
            MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, false, true, 32, true);
        }
        MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, false, true, 32, false);
    }

    if (data_stochastic_rounding && scale_stochastic_rounding) {
        MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, true, false, 16, true);
    }
    if (data_stochastic_rounding) {
        MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, true, false, false, 16, true);
    }
    if (scale_stochastic_rounding) {
        MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, true, false, 16, true);
    }
    MXFP4_DISPATCH_GROUP_CASE(QuantMode::RTE, false, false, false, 16, true);
#undef MXFP4_DISPATCH_GROUP_CASE
}


// ═══════════════════════════════════════════════════════════════════
// Group quantize dim0 — single kernel launch for all groups
//
// Uses mxfp4_v3_fused_group_kernel: one kernel launch over the
// entire contiguous input, with per-group scale pointers in GroupArgs.
// FP4 output is contiguous, scales are per-group.
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_group_quantize_dim0(torch::Tensor input, std::vector<int64_t> group_sizes) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % 128 == 0);

    int ng = (int)group_sizes.size();
    TORCH_CHECK(ng >= 1 && ng <= MAX_GROUPS);

    int64_t total = 0;
    for (auto s : group_sizes) {
        TORCH_CHECK(s % 128 == 0);
        total += s;
    }
    TORCH_CHECK(total == M);

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate contiguous FP4 output
    auto fp4_all = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));

    // Allocate per-group scale tensors
    std::vector<torch::Tensor> sc_allocs(ng);
    GroupArgs args;
    memset(&args, 0, sizeof(args));
    args.num_groups = ng;
    args.boundaries[0] = 0;

    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        args.boundaries[i + 1] = args.boundaries[i] + (int)Mi;
        sc_allocs[i] = torch::empty({Mi / 128, K / 128, 32, 16},
            torch::dtype(torch::kUInt8).device(device));
        args.scale_ptrs[i] = reinterpret_cast<uint8_t*>(sc_allocs[i].data_ptr());
    }

    const int tiles_X = K / CHUNK_DIM;
    const int tiles_Y = M / CHUNK_DIM;

    const auto& ci = get_cached();
    const int dshmem = ci.dshmem;

    // Set up grouped kernel occupancy (first time only)
    static DeviceInitFlags grp_init;
    ensure_device_init(grp_init, [&] {
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v3_fused_group_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    // TMA maps
    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_all.data_ptr(),
                  M, K, TILE_DIM, TILE_DIM, K, 4);

    // Single kernel launch for all groups
    dim3 grid(tiles_X, tiles_Y);
    mxfp4_v3_fused_group_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
        tma_in, tma_out, M, K, args);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v3 group quantize: ", cudaGetErrorString(err));

    // Split FP4 output and pair with per-group scales
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    int64_t row_off = 0;
    for (int i = 0; i < ng; ++i) {
        int64_t Mi = group_sizes[i];
        auto fp4_slice = fp4_all.narrow(0, row_off, Mi).contiguous();
        results.push_back(std::make_tuple(fp4_slice, sc_allocs[i]));
        row_off += Mi;
    }
    return results;
}

// ═══════════════════════════════════════════════════════════════════
// Multi quantize — multiple non-contiguous tensors
// ═══════════════════════════════════════════════════════════════════
std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_multi_quantize(std::vector<torch::Tensor> inputs) {
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    for (auto& t : inputs) {
        results.push_back(mxfp4_quantize_for_gemm(t, 0));
    }
    return results;
}

std::vector<std::tuple<torch::Tensor, torch::Tensor>>
mxfp4_multi_quantize_opt(
    std::vector<torch::Tensor> inputs,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> results;
    results.reserve(inputs.size());
    uint64_t subseq = rng_subsequence;
    for (auto& t : inputs) {
        if (use_rht) {
            results.push_back(mxfp4_quantize_for_gemm_opt_rht(
                t, 0, data_stochastic_rounding, scale_stochastic_rounding,
                rht_block_size, with_random_sign_mask, rng_seed, subseq));
        } else {
            results.push_back(mxfp4_quantize_for_gemm_opt(
                t, 0, data_stochastic_rounding, scale_stochastic_rounding,
                rng_seed, subseq));
        }
        subseq += 1;
    }
    return results;
}


// ═══════════════════════════════════════════════════════════════════
template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_col_only_impl(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fp4_out = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_out.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_col_only_shmem_size();
    static DeviceInitFlags col_init;
    ensure_device_init(col_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_col_only_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_col_only_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_col_only_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(M / CHUNK_DIM, K / CHUNK_DIM);
    mxfp4_v4_col_only_fused_kernel<MODE><<<grid, THREADS, dshmem, stream>>>(
        tma_in, tma_out, reinterpret_cast<uint8_t*>(sc_out.data_ptr()), M, K);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v4 col-only quantize: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_col_only(torch::Tensor input, int mode) {
    switch (mode) {
        case 1: return mxfp4_quantize_col_only_impl<QuantMode::ENCODE>(input);
        case 2: return mxfp4_quantize_col_only_impl<QuantMode::DECODE>(input);
        default: return mxfp4_quantize_col_only_impl<QuantMode::RTE>(input);
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_col_only_opt_impl(
    torch::Tensor input,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
    }
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    if constexpr (WITH_RHT) {
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto fp4_out = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto sc_out = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_out, fp4_out.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_col_only_shmem_size();
    static DeviceInitFlags col_only_opt_init;
    ensure_device_init(col_only_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_col_only_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(M / CHUNK_DIM, K / CHUNK_DIM);
    mxfp4_v4_col_only_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in, tma_out, reinterpret_cast<uint8_t*>(sc_out.data_ptr()),
            M, K, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4_v4 col-only quantize opt: ", cudaGetErrorString(err));
    return std::make_tuple(fp4_out, sc_out);
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_col_only_opt(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    if (!data_stochastic_rounding && !scale_stochastic_rounding) {
        return mxfp4_quantize_col_only(input, mode);
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: return mxfp4_quantize_col_only_opt_impl<QuantMode::ENCODE, true, true>(input, rng_seed, rng_subsequence);
            case 2: return mxfp4_quantize_col_only_opt_impl<QuantMode::DECODE, true, true>(input, rng_seed, rng_subsequence);
            default: return mxfp4_quantize_col_only_opt_impl<QuantMode::RTE, true, true>(input, rng_seed, rng_subsequence);
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: return mxfp4_quantize_col_only_opt_impl<QuantMode::ENCODE, true, false>(input, rng_seed, rng_subsequence);
            case 2: return mxfp4_quantize_col_only_opt_impl<QuantMode::DECODE, true, false>(input, rng_seed, rng_subsequence);
            default: return mxfp4_quantize_col_only_opt_impl<QuantMode::RTE, true, false>(input, rng_seed, rng_subsequence);
        }
    }
    switch (mode) {
        case 1: return mxfp4_quantize_col_only_opt_impl<QuantMode::ENCODE, false, true>(input, rng_seed, rng_subsequence);
        case 2: return mxfp4_quantize_col_only_opt_impl<QuantMode::DECODE, false, true>(input, rng_seed, rng_subsequence);
        default: return mxfp4_quantize_col_only_opt_impl<QuantMode::RTE, false, true>(input, rng_seed, rng_subsequence);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_quantize_col_only_opt_rht(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define COL_ONLY_RHT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_col_only_opt_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, true, RHT_BLOCK, SIGN_FLAG>( \
        input, rng_seed, rng_subsequence)

    if (rht_block_size == 16) {
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, true, 16, true);
                    case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, true, 16, true);
                    default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, true, 16, true);
                }
            }
            if (data_stochastic_rounding) {
                switch (mode) {
                    case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, false, 16, true);
                    case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, false, 16, true);
                    default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, false, 16, true);
                }
            }
            if (scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, true, 16, true);
                    case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, true, 16, true);
                    default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, true, 16, true);
                }
            }
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, false, 16, true);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, false, 16, true);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, false, 16, true);
            }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, true, 16, false);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, true, 16, false);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, true, 16, false);
            }
        }
        if (data_stochastic_rounding) {
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, false, 16, false);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, false, 16, false);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, false, 16, false);
            }
        }
        if (scale_stochastic_rounding) {
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, true, 16, false);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, true, 16, false);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, true, 16, false);
            }
        }
        switch (mode) {
            case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, false, 16, false);
            case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, false, 16, false);
            default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, false, 16, false);
        }
    }

    if (with_random_sign_mask) {
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, true, 32, true);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, true, 32, true);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, true, 32, true);
            }
        }
        if (data_stochastic_rounding) {
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, false, 32, true);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, false, 32, true);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, false, 32, true);
            }
        }
        if (scale_stochastic_rounding) {
            switch (mode) {
                case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, true, 32, true);
                case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, true, 32, true);
                default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, true, 32, true);
            }
        }
        switch (mode) {
            case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, false, 32, true);
            case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, false, 32, true);
            default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, false, 32, true);
        }
    }

    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, true, 32, false);
            case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, true, 32, false);
            default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, true, 32, false);
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, true, false, 32, false);
            case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, true, false, 32, false);
            default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, true, false, 32, false);
        }
    }
    if (scale_stochastic_rounding) {
        switch (mode) {
            case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, true, 32, false);
            case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, true, 32, false);
            default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, true, 32, false);
        }
    }
    switch (mode) {
        case 1: return COL_ONLY_RHT_DISPATCH(QuantMode::ENCODE, false, false, 32, false);
        case 2: return COL_ONLY_RHT_DISPATCH(QuantMode::DECODE, false, false, 32, false);
        default: return COL_ONLY_RHT_DISPATCH(QuantMode::RTE, false, false, 32, false);
    }
#undef COL_ONLY_RHT_DISPATCH
}


// ═══════════════════════════════════════════════════════════════════
// Row + Col quantize: quantize input and emit the transpose contract directly
//
// Returns: (row_fp4, row_sc, col_fp4, col_sc)
// where col is the MXFP4 quantization of input^T
// ═══════════════════════════════════════════════════════════════════
template<bool SHARED_2D_WEIGHT = false>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_impl(torch::Tensor input, int mode) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags rowcol_init;
    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    if constexpr (SHARED_2D_WEIGHT) {
        TORCH_CHECK(mode == 1, "shared 2D weights require encode-centric scaling");
        ensure_device_init(rowcol_init, [&] {
            cudaFuncSetAttribute(
                mxfp4_v4_rowcol_fused_kernel<QuantMode::ENCODE, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        mxfp4_v4_rowcol_fused_kernel<QuantMode::ENCODE, true><<<grid, THREADS, dshmem, stream>>>(
            tma_in,
            tma_row_out,
            tma_col_out,
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            M,
            K);
    } else {
        ensure_device_init(rowcol_init, [&] {
            cudaFuncSetAttribute(mxfp4_v4_rowcol_fused_kernel<QuantMode::RTE, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_rowcol_fused_kernel<QuantMode::ENCODE, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_rowcol_fused_kernel<QuantMode::DECODE, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        switch (mode) {
        case 1:
            mxfp4_v4_rowcol_fused_kernel<QuantMode::ENCODE, SHARED_2D_WEIGHT><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        case 2:
            mxfp4_v4_rowcol_fused_kernel<QuantMode::DECODE, SHARED_2D_WEIGHT><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        default:
            mxfp4_v4_rowcol_fused_kernel<QuantMode::RTE, SHARED_2D_WEIGHT><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 row/col quantize: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col(torch::Tensor input, int mode) {
    return mxfp4_quantize_row_and_col_impl<false>(input, mode);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_weight_2d(torch::Tensor input) {
    return mxfp4_quantize_row_and_col_impl<true>(input, 1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_nhsd_wo_row_and_col(torch::Tensor input, int mode) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
                "NHSD WO MXFP4 quantize expects a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 4,
                "NHSD WO MXFP4 quantize expects bf16 [B,H,S,D]");

    const int64_t B = input.size(0);
    const int64_t H = input.size(1);
    const int64_t S = input.size(2);
    const int64_t D = input.size(3);
    const int64_t M = B * S;
    const int64_t K = H * D;
    TORCH_CHECK(D % TILE_DIM == 0,
                "NHSD WO MXFP4 quantize requires head_dim to be a multiple of 64");
    TORCH_CHECK(S % CHUNK_DIM == 0,
                "NHSD WO MXFP4 quantize requires sequence length to be a multiple of 128");
    TORCH_CHECK(M % CHUNK_DIM == 0 && K % CHUNK_DIM == 0,
                "logical [B*S,H*D] dimensions must be multiples of 128");
    TORCH_CHECK(B <= INT_MAX && H <= INT_MAX && S <= INT_MAX && D <= INT_MAX,
                "NHSD WO MXFP4 quantize dimensions exceed int range");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), B * H * S, D, TILE_DIM, TILE_DIM, D, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags nhsd_wo_init;
    ensure_device_init(nhsd_wo_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_nhsd_wo_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_nhsd_wo_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_nhsd_wo_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_nhsd_wo_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in, tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K, static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D));
            break;
        case 2:
            mxfp4_v4_nhsd_wo_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in, tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K, static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D));
            break;
        default:
            mxfp4_v4_nhsd_wo_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in, tma_row_out, tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K, static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D));
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 NHSD WO row/col quantize: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

template<
    QuantMode MODE,
    bool DATA_SR,
    bool SCALE_SR,
    bool WITH_RHT = false,
    int RHT_BLOCK_SIZE = 16,
    bool WITH_RANDOM_SIGN_MASK = true,
    bool ROW_WITH_RHT = false,
    bool COL_WITH_RHT = WITH_RHT>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_opt_impl(
    torch::Tensor input,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    if constexpr (ROW_WITH_RHT || COL_WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
    }
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    if constexpr (ROW_WITH_RHT) {
        TORCH_CHECK(K % RHT_BLOCK_SIZE == 0, "K must be divisible by the RHT block size for row-output RHT");
    }
    if constexpr (COL_WITH_RHT) {
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor rng_state;
    const uint64_t* rng_state_ptr = nullptr;
    if constexpr (DATA_SR || SCALE_SR || (WITH_RHT && WITH_RANDOM_SIGN_MASK)) {
        rng_state = make_mxfp4_forward_advancing_rng_state(
            input, rng_seed, rng_subsequence, stream.stream());
        rng_state_ptr = reinterpret_cast<const uint64_t*>(
            rng_state.data_ptr<int64_t>());
    }

    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags rowcol_opt_init;
    ensure_device_init(rowcol_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_rowcol_fused_kernel_opt<
                MODE,
                DATA_SR,
                SCALE_SR,
                WITH_RHT,
                RHT_BLOCK_SIZE,
                WITH_RANDOM_SIGN_MASK,
                ROW_WITH_RHT,
                COL_WITH_RHT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_rowcol_fused_kernel_opt<
        MODE,
        DATA_SR,
        SCALE_SR,
        WITH_RHT,
        RHT_BLOCK_SIZE,
        WITH_RANDOM_SIGN_MASK,
        ROW_WITH_RHT,
        COL_WITH_RHT>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in,
            tma_row_out,
            tma_col_out,
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            M,
            K,
            rng_seed,
            rng_subsequence,
            rng_state_ptr);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 row/col quantize opt: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_opt(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    if (!data_stochastic_rounding && !scale_stochastic_rounding) {
        return mxfp4_quantize_row_and_col(input, mode);
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::ENCODE, true, true>(input, rng_seed, rng_subsequence);
            case 2: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::DECODE, true, true>(input, rng_seed, rng_subsequence);
            default: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::RTE, true, true>(input, rng_seed, rng_subsequence);
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::ENCODE, true, false>(input, rng_seed, rng_subsequence);
            case 2: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::DECODE, true, false>(input, rng_seed, rng_subsequence);
            default: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::RTE, true, false>(input, rng_seed, rng_subsequence);
        }
    }
    switch (mode) {
        case 1: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::ENCODE, false, true>(input, rng_seed, rng_subsequence);
        case 2: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::DECODE, false, true>(input, rng_seed, rng_subsequence);
        default: return mxfp4_quantize_row_and_col_opt_impl<QuantMode::RTE, false, true>(input, rng_seed, rng_subsequence);
    }
}

template<bool ROW_WITH_RHT, bool COL_WITH_RHT>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_opt_rht_axes(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define ROWCOL_RHT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_row_and_col_opt_impl< \
        MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, true, RHT_BLOCK, SIGN_FLAG, ROW_WITH_RHT, COL_WITH_RHT>( \
        input, rng_seed, rng_subsequence)
    if (rht_block_size == 16) {
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, true, 16, true);
                    case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, true, 16, true);
                    default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, true, 16, true);
                }
            }
            if (data_stochastic_rounding) {
                switch (mode) {
                    case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, false, 16, true);
                    case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, false, 16, true);
                    default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, false, 16, true);
                }
            }
            if (scale_stochastic_rounding) {
                switch (mode) {
                    case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, true, 16, true);
                    case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, true, 16, true);
                    default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, true, 16, true);
                }
            }
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, false, 16, true);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, false, 16, true);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, false, 16, true);
            }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, true, 16, false);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, true, 16, false);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, true, 16, false);
            }
        }
        if (data_stochastic_rounding) {
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, false, 16, false);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, false, 16, false);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, false, 16, false);
            }
        }
        if (scale_stochastic_rounding) {
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, true, 16, false);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, true, 16, false);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, true, 16, false);
            }
        }
        switch (mode) {
            case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, false, 16, false);
            case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, false, 16, false);
            default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, false, 16, false);
        }
    }
    if (with_random_sign_mask) {
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, true, 32, true);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, true, 32, true);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, true, 32, true);
            }
        }
        if (data_stochastic_rounding) {
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, false, 32, true);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, false, 32, true);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, false, 32, true);
            }
        }
        if (scale_stochastic_rounding) {
            switch (mode) {
                case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, true, 32, true);
                case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, true, 32, true);
                default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, true, 32, true);
            }
        }
        switch (mode) {
            case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, false, 32, true);
            case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, false, 32, true);
            default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, false, 32, true);
        }
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, true, 32, false);
            case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, true, 32, false);
            default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, true, 32, false);
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, true, false, 32, false);
            case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, true, false, 32, false);
            default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, true, false, 32, false);
        }
    }
    if (scale_stochastic_rounding) {
        switch (mode) {
            case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, true, 32, false);
            case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, true, 32, false);
            default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, true, 32, false);
        }
    }
    switch (mode) {
        case 1: return ROWCOL_RHT_DISPATCH(QuantMode::ENCODE, false, false, 32, false);
        case 2: return ROWCOL_RHT_DISPATCH(QuantMode::DECODE, false, false, 32, false);
        default: return ROWCOL_RHT_DISPATCH(QuantMode::RTE, false, false, 32, false);
    }
#undef ROWCOL_RHT_DISPATCH
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_opt_rht(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    return mxfp4_quantize_row_and_col_opt_rht_axes<false, true>(
        input,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_opt_rht_both(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    return mxfp4_quantize_row_and_col_opt_rht_axes<true, true>(
        input,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_row_and_col_opt_rht_row_only(
    torch::Tensor input,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    return mxfp4_quantize_row_and_col_opt_rht_axes<true, false>(
        input,
        mode,
        data_stochastic_rounding,
        scale_stochastic_rounding,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence);
}


void mxfp4_quantize_row_and_col_launch_inplace(
    torch::Tensor input,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags rowcol_init;
    ensure_device_init(rowcol_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_rowcol_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_rowcol_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_rowcol_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_rowcol_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        case 2:
            mxfp4_v4_rowcol_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
        default:
            mxfp4_v4_rowcol_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                K);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 row/col quantize: ", cudaGetErrorString(err));
}


void mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {h1_raw, h3}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(),
                "h1_raw and h3 must have the same shape");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, h1_raw.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags fused_silu_mul_init;
    ensure_device_init(fused_silu_mul_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_mul_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H);
            break;
        case 2:
            mxfp4_v4_fused_silu_mul_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H);
            break;
        default:
            mxfp4_v4_fused_silu_mul_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu mul row/col quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor sig_h1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {h1_raw, h3}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(),
                "h1_raw and h3 must have the same shape");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(sig_h1.is_cuda() && sig_h1.is_contiguous());
    TORCH_CHECK(sig_h1.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(sig_h1.sizes() == torch::IntArrayRef({M, H}));
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags fused_silu_mul_sigmoid_init;
    ensure_device_init(fused_silu_mul_sigmoid_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<IType*>(sig_h1.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H);
            break;
        case 2:
            mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<IType*>(sig_h1.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H);
            break;
        default:
            mxfp4_v4_fused_silu_mul_sigmoid_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<IType*>(sig_h1.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu mul sigmoid row/col quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace(
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h1_raw.size(0) % 128 == 0 && h1_raw.size(1) % 128 == 0,
                "input must be a contiguous bf16 matrix with dims divisible by 128");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, h1_raw.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const bool use_tma = env_enabled("MXFP4_USE_TMA_SQRELU_QUANT");
    const int dshmem = v4_rowcol_shmem_size();
    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);

    if (use_tma) {
        static DeviceInitFlags fused_sqrelu_tma_init;
        ensure_device_init(fused_sqrelu_tma_init, [&] {
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_rowcol_tma_kernel<QuantMode::RTE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_rowcol_tma_kernel<QuantMode::ENCODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_rowcol_tma_kernel<QuantMode::DECODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        switch (mode) {
            case 1:
                mxfp4_v4_fused_sqrelu_rowcol_tma_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                    tma_in, tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            case 2:
                mxfp4_v4_fused_sqrelu_rowcol_tma_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                    tma_in, tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            default:
                mxfp4_v4_fused_sqrelu_rowcol_tma_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                    tma_in, tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
        }
    } else {
        static DeviceInitFlags fused_sqrelu_direct_init;
        ensure_device_init(fused_sqrelu_direct_init, [&] {
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_rowcol_kernel<QuantMode::RTE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_rowcol_kernel<QuantMode::ENCODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_rowcol_kernel<QuantMode::DECODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        switch (mode) {
            case 1:
                mxfp4_v4_fused_sqrelu_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            case 2:
                mxfp4_v4_fused_sqrelu_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            default:
                mxfp4_v4_fused_sqrelu_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused sqrelu row/col quantize: ", cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_sqrelu_quantize_row_and_col(
    torch::Tensor h1_raw,
    int mode
) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);

    int64_t M = h1_raw.size(0), H = h1_raw.size(1);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0, "M and H must be multiples of 128");

    auto device = h1_raw.device();
    auto row_fp4 = torch::empty({M, H / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, H / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace(
        h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

void mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {dh, h1_raw}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_dh{}, tma_h1{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_dh, dh.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_h1, h1_raw.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const bool use_tma = env_enabled("MXFP4_USE_TMA_SQRELU_QUANT");
    const int dshmem = use_tma ? v4_dual_input_rowcol_shmem_size() : v4_rowcol_shmem_size();
    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);

    if (use_tma) {
        static DeviceInitFlags fused_sqrelu_deriv_tma_init;
        ensure_device_init(fused_sqrelu_deriv_tma_init, [&] {
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel<QuantMode::RTE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel<QuantMode::ENCODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel<QuantMode::DECODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        switch (mode) {
            case 1:
                mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                    tma_dh, tma_h1, tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            case 2:
                mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                    tma_dh, tma_h1, tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            default:
                mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                    tma_dh, tma_h1, tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
        }
    } else {
        static DeviceInitFlags fused_sqrelu_deriv_direct_init;
        ensure_device_init(fused_sqrelu_deriv_direct_init, [&] {
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel<QuantMode::RTE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel<QuantMode::ENCODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            cudaFuncSetAttribute(mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel<QuantMode::DECODE>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        switch (mode) {
            case 1:
                mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(dh.data_ptr()),
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            case 2:
                mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(dh.data_ptr()),
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
            default:
                mxfp4_v4_fused_sqrelu_deriv_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(dh.data_ptr()),
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    tma_row_out, tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()), M, H);
                break;
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused sqrelu deriv row/col quantize: ", cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_sqrelu_deriv_quantize_row_and_col(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    int mode
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have the same shape");

    int64_t M = dh.size(0), H = dh.size(1);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0, "M and H must be multiples of 128");

    auto device = dh.device();
    auto row_fp4 = torch::empty({M, H / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, H / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace(
        dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

template<bool ROW_WITH_RHT, bool COL_WITH_RHT>
void mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace_axes(
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(mode == 1, "fused square-ReLU opt quantize currently supports mode=1/ENCODE");
    TORCH_CHECK(!(data_stochastic_rounding && scale_stochastic_rounding),
                "MXFP4 fused square-ReLU opt quantize requires data SR and scale SR to be mutually exclusive");
    TORCH_CHECK(!with_random_sign_mask,
                "MXFP4 fused square-ReLU opt quantize currently requires with_random_sign_mask=false");
    if constexpr (ROW_WITH_RHT || COL_WITH_RHT) {
        TORCH_CHECK(rht_block_size == 16 || rht_block_size == 32, "RHT block size must be 16 or 32");
    }
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h1_raw.size(0) % 128 == 0 && h1_raw.size(1) % 128 == 0,
                "input must be a contiguous bf16 matrix with dims divisible by 128");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, h1_raw.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);

    TORCH_CHECK(rht_block_size == 32, "MXFP4 fused square-ReLU opt quantize currently supports rht_block_size=32");
    TORCH_CHECK(!ROW_WITH_RHT, "MXFP4 fused square-ReLU opt quantize currently supports no row-RHT");
    TORCH_CHECK(!scale_stochastic_rounding, "MXFP4 fused square-ReLU opt quantize currently supports data SR only");

    if constexpr (COL_WITH_RHT) {
        TORCH_CHECK(!data_stochastic_rounding,
                    "MXFP4 fused square-ReLU activation col-RHT path currently supports no data SR");
        static DeviceInitFlags fused_sqrelu_opt_col_rht_init;
        ensure_device_init(fused_sqrelu_opt_col_rht_init, [&] {
            cudaFuncSetAttribute(
                mxfp4_v4_fused_sqrelu_rowcol_tma_kernel_opt<QuantMode::ENCODE, false, false, true, 32, false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        mxfp4_v4_fused_sqrelu_rowcol_tma_kernel_opt<QuantMode::ENCODE, false, false, true, 32, false, false, true>
            <<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, H, rng_seed, rng_subsequence);
    } else {
        TORCH_CHECK(data_stochastic_rounding,
                    "MXFP4 fused square-ReLU no-RHT opt path is only compiled for data SR");
        static DeviceInitFlags fused_sqrelu_opt_data_sr_init;
        ensure_device_init(fused_sqrelu_opt_data_sr_init, [&] {
            cudaFuncSetAttribute(
                mxfp4_v4_fused_sqrelu_rowcol_tma_kernel_opt<QuantMode::ENCODE, true, false, false, 32, false, false, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        mxfp4_v4_fused_sqrelu_rowcol_tma_kernel_opt<QuantMode::ENCODE, true, false, false, 32, false, false, false>
            <<<grid, THREADS, dshmem, stream>>>(
                tma_in,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, H, rng_seed, rng_subsequence);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused sqrelu opt row/col quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace(
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    bool row_with_rht = false
) {
    if (row_with_rht && use_rht) {
        return mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace_axes<true, true>(
            h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (row_with_rht) {
        return mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace_axes<true, false>(
            h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (use_rht) {
        return mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace_axes<false, true>(
            h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (data_stochastic_rounding || scale_stochastic_rounding) {
        return mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace_axes<false, false>(
            h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    return mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace(
        h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode);
}

template<bool ROW_WITH_RHT, bool COL_WITH_RHT>
void mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace_axes(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(mode == 1, "fused square-ReLU derivative opt quantize currently supports mode=1/ENCODE");
    TORCH_CHECK(!(data_stochastic_rounding && scale_stochastic_rounding),
                "MXFP4 fused square-ReLU derivative opt quantize requires data SR and scale SR to be mutually exclusive");
    TORCH_CHECK(!with_random_sign_mask,
                "MXFP4 fused square-ReLU derivative opt quantize currently requires with_random_sign_mask=false");
    if constexpr (ROW_WITH_RHT || COL_WITH_RHT) {
        TORCH_CHECK(rht_block_size == 16 || rht_block_size == 32, "RHT block size must be 16 or 32");
    }
    for (const auto& input : {dh, h1_raw}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h1_raw.sizes(), "dh and h1_raw must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_dh{}, tma_h1{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_dh, dh.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_h1, h1_raw.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_dual_input_rowcol_shmem_size();
    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);

    TORCH_CHECK(rht_block_size == 32,
                "MXFP4 fused square-ReLU derivative opt quantize currently supports rht_block_size=32");
    TORCH_CHECK(!ROW_WITH_RHT,
                "MXFP4 fused square-ReLU derivative opt quantize currently supports no row-RHT");
    TORCH_CHECK(!scale_stochastic_rounding,
                "MXFP4 fused square-ReLU derivative opt quantize currently supports data SR only");
    TORCH_CHECK(data_stochastic_rounding || COL_WITH_RHT,
                "MXFP4 fused square-ReLU derivative opt quantize requires data SR or col-RHT");

    if constexpr (COL_WITH_RHT) {
        if (data_stochastic_rounding) {
            static DeviceInitFlags fused_sqrelu_deriv_opt_col_rht_data_sr_init;
            ensure_device_init(fused_sqrelu_deriv_opt_col_rht_data_sr_init, [&] {
                cudaFuncSetAttribute(
                    mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt<QuantMode::ENCODE, true, false, true, 32, false, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            });
            mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt<QuantMode::ENCODE, true, false, true, 32, false, false, true>
                <<<grid, THREADS, dshmem, stream>>>(
                    tma_dh,
                    tma_h1,
                    tma_row_out,
                    tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                    M, H, rng_seed, rng_subsequence);
        } else {
            static DeviceInitFlags fused_sqrelu_deriv_opt_col_rht_init;
            ensure_device_init(fused_sqrelu_deriv_opt_col_rht_init, [&] {
                cudaFuncSetAttribute(
                    mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt<QuantMode::ENCODE, false, false, true, 32, false, false, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
            });
            mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt<QuantMode::ENCODE, false, false, true, 32, false, false, true>
                <<<grid, THREADS, dshmem, stream>>>(
                    tma_dh,
                    tma_h1,
                    tma_row_out,
                    tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                    M, H, rng_seed, rng_subsequence);
        }
    } else {
        TORCH_CHECK(data_stochastic_rounding,
                    "MXFP4 fused square-ReLU derivative no-RHT opt path is only compiled for data SR");
        static DeviceInitFlags fused_sqrelu_deriv_opt_data_sr_init;
        ensure_device_init(fused_sqrelu_deriv_opt_data_sr_init, [&] {
            cudaFuncSetAttribute(
                mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt<QuantMode::ENCODE, true, false, false, 32, false, false, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        });
        mxfp4_v4_fused_sqrelu_deriv_rowcol_tma_kernel_opt<QuantMode::ENCODE, true, false, false, 32, false, false, false>
            <<<grid, THREADS, dshmem, stream>>>(
                tma_dh,
                tma_h1,
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, H, rng_seed, rng_subsequence);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused sqrelu deriv opt row/col quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    bool row_with_rht = false
) {
    if (row_with_rht && use_rht) {
        return mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace_axes<true, true>(
            dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (row_with_rht) {
        return mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace_axes<true, false>(
            dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (use_rht) {
        return mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace_axes<false, true>(
            dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (data_stochastic_rounding || scale_stochastic_rounding) {
        return mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace_axes<false, false>(
            dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    return mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace(
        dh, h1_raw, row_fp4, row_sc, col_fp4, col_sc, mode);
}

void mxfp4_fused_silu_mul_quantize_row_and_col_strided_launch_inplace(
    torch::Tensor h13,
    int64_t H,
    int64_t h3_offset,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);
    TORCH_CHECK(H > 0 && H % 128 == 0, "H must be positive and divisible by 128");
    TORCH_CHECK(h3_offset >= H, "h3_offset must place h3 after h1");
    const int64_t M = h13.size(0);
    const int64_t input_stride = h13.size(1);
    TORCH_CHECK(M % 128 == 0, "M must be divisible by 128");
    TORCH_CHECK(h3_offset + H <= input_stride, "h13 does not contain h3 slice");

    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags fused_silu_mul_strided_init;
    ensure_device_init(fused_silu_mul_strided_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_strided_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_strided_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_strided_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_mul_rowcol_strided_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h13.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H,
                input_stride,
                h3_offset);
            break;
        case 2:
            mxfp4_v4_fused_silu_mul_rowcol_strided_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h13.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H,
                input_stride,
                h3_offset);
            break;
        default:
            mxfp4_v4_fused_silu_mul_rowcol_strided_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                reinterpret_cast<const IType*>(h13.data_ptr()),
                tma_row_out,
                tma_col_out,
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M,
                H,
                input_stride,
                h3_offset);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused strided silu mul row/col quantize: ", cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_silu_mul_quantize_row_and_col_strided(
    torch::Tensor h13,
    int64_t H,
    int64_t h3_offset,
    int mode
) {
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);
    int64_t M = h13.size(0);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0, "M and H must be multiples of 128");

    auto device = h13.device();
    auto row_fp4 = torch::empty({M, H / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, H / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    mxfp4_fused_silu_mul_quantize_row_and_col_strided_launch_inplace(
        h13, H, h3_offset, row_fp4, row_sc, col_fp4, col_sc, mode);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

void mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only_launch_inplace(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(rht_block_size == 32, "fused SiLU row-RHT path currently supports rht_block_size=32");
    TORCH_CHECK(!with_random_sign_mask, "fused SiLU row-RHT path currently requires with_random_sign_mask=false");
    for (const auto& input : {h1_raw, h3}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(),
                "h1_raw and h3 must have the same shape");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags fused_silu_mul_rht_row_init;
    ensure_device_init(fused_silu_mul_rht_row_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<QuantMode::RTE, false, false, true, false, 32, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<QuantMode::ENCODE, false, false, true, false, 32, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<QuantMode::DECODE, false, false, true, false, 32, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<QuantMode::ENCODE, false, false, true, false, 32, false>
                <<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    reinterpret_cast<const IType*>(h3.data_ptr()),
                    tma_row_out,
                    tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                    M,
                    H,
                    rng_seed,
                    rng_subsequence);
            break;
        case 2:
            mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<QuantMode::DECODE, false, false, true, false, 32, false>
                <<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    reinterpret_cast<const IType*>(h3.data_ptr()),
                    tma_row_out,
                    tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                    M,
                    H,
                    rng_seed,
                    rng_subsequence);
            break;
        default:
            mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<QuantMode::RTE, false, false, true, false, 32, false>
                <<<grid, THREADS, dshmem, stream>>>(
                    reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                    reinterpret_cast<const IType*>(h3.data_ptr()),
                    tma_row_out,
                    tma_col_out,
                    reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                    reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                    M,
                    H,
                    rng_seed,
                    rng_subsequence);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu mul row-RHT row/col quantize: ", cudaGetErrorString(err));
}

template<bool ROW_WITH_RHT, bool COL_WITH_RHT>
void mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace_axes(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {h1_raw, h3}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(),
                "h1_raw and h3 must have the same shape");

    const int64_t M = h1_raw.size(0);
    const int64_t H = h1_raw.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));
    if constexpr (ROW_WITH_RHT || COL_WITH_RHT) {
        TORCH_CHECK(rht_block_size == 16 || rht_block_size == 32, "RHT block size must be 16 or 32");
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);

#define FUSED_SILU_OPT_LAUNCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, RHT_BLOCK, SIGN_FLAG) \
    do { \
        static DeviceInitFlags fused_silu_opt_init; \
        ensure_device_init(fused_silu_opt_init, [&] { \
            cudaFuncSetAttribute( \
                mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, ROW_WITH_RHT, COL_WITH_RHT, RHT_BLOCK, SIGN_FLAG>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem); \
        }); \
        mxfp4_v4_fused_silu_mul_rowcol_kernel_opt<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, ROW_WITH_RHT, COL_WITH_RHT, RHT_BLOCK, SIGN_FLAG> \
            <<<grid, THREADS, dshmem, stream>>>( \
                reinterpret_cast<const IType*>(h1_raw.data_ptr()), \
                reinterpret_cast<const IType*>(h3.data_ptr()), \
                tma_row_out, \
                tma_col_out, \
                reinterpret_cast<uint8_t*>(row_sc.data_ptr()), \
                reinterpret_cast<uint8_t*>(col_sc.data_ptr()), \
                M, H, rng_seed, rng_subsequence); \
    } while (0)

#define FUSED_SILU_OPT_DISPATCH(DATA_SR_FLAG, SCALE_SR_FLAG) \
    do { \
        if (rht_block_size == 16) { \
            if (with_random_sign_mask) { \
                switch (mode) { \
                    case 1: FUSED_SILU_OPT_LAUNCH(QuantMode::ENCODE, DATA_SR_FLAG, SCALE_SR_FLAG, 16, true); break; \
                    case 2: FUSED_SILU_OPT_LAUNCH(QuantMode::DECODE, DATA_SR_FLAG, SCALE_SR_FLAG, 16, true); break; \
                    default: FUSED_SILU_OPT_LAUNCH(QuantMode::RTE, DATA_SR_FLAG, SCALE_SR_FLAG, 16, true); break; \
                } \
            } else { \
                switch (mode) { \
                    case 1: FUSED_SILU_OPT_LAUNCH(QuantMode::ENCODE, DATA_SR_FLAG, SCALE_SR_FLAG, 16, false); break; \
                    case 2: FUSED_SILU_OPT_LAUNCH(QuantMode::DECODE, DATA_SR_FLAG, SCALE_SR_FLAG, 16, false); break; \
                    default: FUSED_SILU_OPT_LAUNCH(QuantMode::RTE, DATA_SR_FLAG, SCALE_SR_FLAG, 16, false); break; \
                } \
            } \
        } else { \
            if (with_random_sign_mask) { \
                switch (mode) { \
                    case 1: FUSED_SILU_OPT_LAUNCH(QuantMode::ENCODE, DATA_SR_FLAG, SCALE_SR_FLAG, 32, true); break; \
                    case 2: FUSED_SILU_OPT_LAUNCH(QuantMode::DECODE, DATA_SR_FLAG, SCALE_SR_FLAG, 32, true); break; \
                    default: FUSED_SILU_OPT_LAUNCH(QuantMode::RTE, DATA_SR_FLAG, SCALE_SR_FLAG, 32, true); break; \
                } \
            } else { \
                switch (mode) { \
                    case 1: FUSED_SILU_OPT_LAUNCH(QuantMode::ENCODE, DATA_SR_FLAG, SCALE_SR_FLAG, 32, false); break; \
                    case 2: FUSED_SILU_OPT_LAUNCH(QuantMode::DECODE, DATA_SR_FLAG, SCALE_SR_FLAG, 32, false); break; \
                    default: FUSED_SILU_OPT_LAUNCH(QuantMode::RTE, DATA_SR_FLAG, SCALE_SR_FLAG, 32, false); break; \
                } \
            } \
        } \
    } while (0)

    if (data_stochastic_rounding && scale_stochastic_rounding) {
        FUSED_SILU_OPT_DISPATCH(true, true);
    } else if (data_stochastic_rounding) {
        FUSED_SILU_OPT_DISPATCH(true, false);
    } else if (scale_stochastic_rounding) {
        FUSED_SILU_OPT_DISPATCH(false, true);
    } else {
        FUSED_SILU_OPT_DISPATCH(false, false);
    }
#undef FUSED_SILU_OPT_DISPATCH
#undef FUSED_SILU_OPT_LAUNCH

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu mul opt row/col quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence,
    bool row_with_rht = false
) {
    if (row_with_rht && use_rht) {
        return mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace_axes<true, true>(
            h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (row_with_rht) {
        return mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace_axes<true, false>(
            h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (use_rht) {
        return mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace_axes<false, true>(
            h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    if (data_stochastic_rounding || scale_stochastic_rounding) {
        return mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace_axes<false, false>(
            h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode,
            data_stochastic_rounding, scale_stochastic_rounding,
            rht_block_size, with_random_sign_mask, rng_seed, rng_subsequence);
    }
    return mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
        h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_silu_mul_quantize_row_and_col(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    int mode
) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have the same shape");

    int64_t M = h1_raw.size(0), H = h1_raw.size(1);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0, "M and H must be multiples of 128");

    auto device = h1_raw.device();
    auto row_fp4 = torch::empty({M, H / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, H / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
        h1_raw, h3, row_fp4, row_sc, col_fp4, col_sc, mode);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only(
    torch::Tensor h1_raw,
    torch::Tensor h3,
    int mode,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16 && h1_raw.dim() == 2);
    TORCH_CHECK(h3.scalar_type() == torch::kBFloat16 && h3.dim() == 2);
    TORCH_CHECK(h1_raw.sizes() == h3.sizes(), "h1_raw and h3 must have the same shape");

    int64_t M = h1_raw.size(0), H = h1_raw.size(1);
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0, "M and H must be multiples of 128");

    auto device = h1_raw.device();
    auto row_fp4 = torch::empty({M, H / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, H / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only_launch_inplace(
        h1_raw,
        h3,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        mode,
        rht_block_size,
        with_random_sign_mask,
        rng_seed,
        rng_subsequence);
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


std::tuple<torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_to_bf16(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16 && norm_weight.dim() == 1);

    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(norm_weight.size(0) == K, "norm_weight must have shape (K,)");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto normed = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
    auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));

    mxfp4_fused_rmsnorm_to_bf16_kernel<RMS_BLOCK_THREADS><<<M, RMS_BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const IType*>(input.data_ptr()),
        reinterpret_cast<const IType*>(norm_weight.data_ptr()),
        reinterpret_cast<IType*>(normed.data_ptr()),
        inv_rms.data_ptr<float>(),
        epsilon,
        static_cast<int>(M),
        static_cast<int>(K));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused rmsnorm to bf16: ", cudaGetErrorString(err));
    return std::make_tuple(normed, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_silu_deriv_quantize_split2_row_and_col(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto device = dh.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty(
        {2, M, H / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {2, M / 128, H / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {2, H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {2, H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_row_out0{}, tma_row_out1{}, tma_col_out0{}, tma_col_out1{};
    create_tma_2d(tma_row_out0, row_fp4[0].data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_row_out1, row_fp4[1].data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out0, col_fp4[0].data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);
    create_tma_2d(tma_col_out1, col_fp4[1].data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (4 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_init;
    ensure_device_init(fused_split2_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out0, reinterpret_cast<uint8_t*>(row_sc[0].data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col_sc[0].data_ptr()),
                tma_row_out1, reinterpret_cast<uint8_t*>(row_sc[1].data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col_sc[1].data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out0, reinterpret_cast<uint8_t*>(row_sc[0].data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col_sc[0].data_ptr()),
                tma_row_out1, reinterpret_cast<uint8_t*>(row_sc[1].data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col_sc[1].data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out0, reinterpret_cast<uint8_t*>(row_sc[0].data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col_sc[0].data_ptr()),
                tma_row_out1, reinterpret_cast<uint8_t*>(row_sc[1].data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col_sc[1].data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu deriv split2 row/col quantize: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


void mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({2, M, H / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({2, M / 128, H / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({2, H, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({2, H / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out0{}, tma_row_out1{}, tma_col_out0{}, tma_col_out1{};
    create_tma_2d(tma_row_out0, row_fp4[0].data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_row_out1, row_fp4[1].data_ptr(), M, H, TILE_DIM, TILE_DIM, H, 4);
    create_tma_2d(tma_col_out0, col_fp4[0].data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);
    create_tma_2d(tma_col_out1, col_fp4[1].data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (4 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_init;
    ensure_device_init(fused_split2_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out0, reinterpret_cast<uint8_t*>(row_sc[0].data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col_sc[0].data_ptr()),
                tma_row_out1, reinterpret_cast<uint8_t*>(row_sc[1].data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col_sc[1].data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out0, reinterpret_cast<uint8_t*>(row_sc[0].data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col_sc[0].data_ptr()),
                tma_row_out1, reinterpret_cast<uint8_t*>(row_sc[1].data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col_sc[1].data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out0, reinterpret_cast<uint8_t*>(row_sc[0].data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col_sc[0].data_ptr()),
                tma_row_out1, reinterpret_cast<uint8_t*>(row_sc[1].data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col_sc[1].data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu deriv split2 row/col quantize: ", cudaGetErrorString(err));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    auto device = dh.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty(
        {M, H},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M / 128, (2 * H) / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col0_fp4 = torch::empty(
        {H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col0_sc = torch::empty(
        {H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col1_fp4 = torch::empty(
        {H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col1_sc = torch::empty(
        {H / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out0{}, tma_col_out1{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);
    create_tma_2d(tma_col_out0, col0_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);
    create_tma_2d(tma_col_out1, col1_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (4 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_splitcols_init;
    ensure_device_init(fused_split2_splitcols_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu deriv split2 splitcols quantize: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc);
}


void mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col0_fp4,
    torch::Tensor col0_sc,
    torch::Tensor col1_fp4,
    torch::Tensor col1_sc,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, (2 * H) / 128, 32, 16}));
    for (const auto& input : {col0_fp4, col1_fp4}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kFloat4_e2m1fn_x2);
        TORCH_CHECK(input.sizes() == torch::IntArrayRef({H, M / 2}));
    }
    for (const auto& input : {col0_sc, col1_sc}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kUInt8);
        TORCH_CHECK(input.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out0{}, tma_col_out1{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);
    create_tma_2d(tma_col_out0, col0_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);
    create_tma_2d(tma_col_out1, col1_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (4 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_splitcols_init;
    ensure_device_init(fused_split2_splitcols_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_splitcols_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu deriv split2 splitcols quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor sig_h1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col0_fp4,
    torch::Tensor col0_sc,
    torch::Tensor col1_fp4,
    torch::Tensor col1_sc,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw, sig_h1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes() && dh.sizes() == sig_h1.sizes(),
                "dh, h3, h1_raw, and sig_h1 must have the same shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, (2 * H) / 128, 32, 16}));
    for (const auto& input : {col0_fp4, col1_fp4}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kFloat4_e2m1fn_x2);
        TORCH_CHECK(input.sizes() == torch::IntArrayRef({H, M / 2}));
    }
    for (const auto& input : {col0_sc, col1_sc}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kUInt8);
        TORCH_CHECK(input.sizes() == torch::IntArrayRef({H / 128, M / 128, 32, 16}));
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out0{}, tma_col_out1{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);
    create_tma_2d(tma_col_out0, col0_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);
    create_tma_2d(tma_col_out1, col1_fp4.data_ptr(), H, M, TILE_DIM, TILE_DIM, M, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (4 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_from_sigmoid_splitcols_init;
    ensure_device_init(fused_split2_from_sigmoid_splitcols_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(sig_h1.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(sig_h1.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_from_sigmoid_split2_rowcol_splitcols_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(sig_h1.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out0, reinterpret_cast<uint8_t*>(col0_sc.data_ptr()),
                tma_col_out1, reinterpret_cast<uint8_t*>(col1_sc.data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused saved-sigmoid silu deriv split2 splitcols quantize: ", cudaGetErrorString(err));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined(
    torch::Tensor dh,
    torch::Tensor h13,
    int64_t H,
    int64_t h3_offset,
    int mode
) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16 && dh.dim() == 2);
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16 && h13.dim() == 2);
    const int64_t M = dh.size(0);
    const int64_t dh_stride = dh.size(1);
    const int64_t h13_stride = h13.size(1);
    TORCH_CHECK(M == h13.size(0), "dh and h13 row count mismatch");
    TORCH_CHECK(M % 128 == 0 && H % 128 == 0, "M and H must be divisible by 128");
    TORCH_CHECK(dh_stride >= H, "dh stride must cover H");
    TORCH_CHECK(h3_offset >= H && h3_offset + H <= h13_stride, "h13 must contain packed h1 and h3 slices");

    auto device = dh.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto row_fp4 = torch::empty(
        {M, H},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M / 128, (2 * H) / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {2 * H, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {(2 * H) / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), 2 * H, M, TILE_DIM, TILE_DIM, M, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (4 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_strided_combined_init;
    ensure_device_init(fused_split2_strided_combined_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h13.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, H, dh_stride, h13_stride, h3_offset);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h13.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, H, dh_stride, h13_stride, h3_offset);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_rowcol_strided_combined_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h13.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, H, dh_stride, h13_stride, h3_offset);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused strided silu deriv split2 row/col combined quantize: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


void mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1_out,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw, dh1_out, dh3_out}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have the same shape");
    TORCH_CHECK(dh1_out.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1_out and dh3_out must match dh shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, (2 * H) / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    static DeviceInitFlags fused_split2_row_bf16_init;
    ensure_device_init(fused_split2_row_bf16_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                nullptr,
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                nullptr,
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                nullptr,
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu deriv split2 row bf16 quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor sig_h1,
    torch::Tensor dh1_out,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw, sig_h1, dh1_out, dh3_out}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes() && dh.sizes() == sig_h1.sizes(),
                "dh, h3, h1_raw, and sig_h1 must have the same shape");
    TORCH_CHECK(dh1_out.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1_out and dh3_out must match dh shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, (2 * H) / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(NUM_TILES * TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(BUFFS_OUT * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT);

    static DeviceInitFlags fused_saved_sigmoid_split2_row_bf16_init;
    ensure_device_init(fused_saved_sigmoid_split2_row_bf16_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::RTE, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::ENCODE, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::DECODE, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::ENCODE, true><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(sig_h1.data_ptr()),
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::DECODE, true><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(sig_h1.data_ptr()),
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_kernel<QuantMode::RTE, true><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<const IType*>(sig_h1.data_ptr()),
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused saved-sigmoid silu deriv split2 row bf16 quantize: ", cudaGetErrorString(err));
}

void mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace(
    torch::Tensor dh,
    torch::Tensor h3,
    torch::Tensor h1_raw,
    torch::Tensor dh1_out,
    torch::Tensor dh3_out,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    int mode
) {
    for (const auto& input : {dh, h3, h1_raw, dh1_out, dh3_out}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes(),
                "dh, h3, and h1_raw must have the same shape");
    TORCH_CHECK(dh1_out.sizes() == dh.sizes() && dh3_out.sizes() == dh.sizes(),
                "dh1_out and dh3_out must match dh shape");

    const int64_t M = dh.size(0);
    const int64_t H = dh.size(1);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, H}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, (2 * H) / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_row_out{};
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, 2 * H, TILE_DIM, TILE_DIM, 2 * H, 4);

    constexpr int direct_dshmem =
        (2 * DIVUP_TO_MULTIPLE(TILE_DIM * TILE_DIM * (int)sizeof(IType), TMA_SHMEM_ALIGNMENT)) +
        DIVUP_TO_MULTIPLE(4 * OUT_SIZE, TMA_SHMEM_ALIGNMENT) +
        (2 * DIVUP_TO_MULTIPLE(CHUNK_DIM * SCALES_PER_CHUNK, TMA_SHMEM_ALIGNMENT));

    static DeviceInitFlags fused_split2_row_bf16_tile_init;
    ensure_device_init(fused_split2_row_bf16_tile_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, direct_dshmem);
    });

    dim3 grid(H / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel<QuantMode::ENCODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
        case 2:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel<QuantMode::DECODE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
        default:
            mxfp4_v4_fused_silu_deriv_split2_row_bf16_tile_kernel<QuantMode::RTE><<<grid, THREADS, direct_dshmem, stream>>>(
                reinterpret_cast<const IType*>(dh.data_ptr()),
                reinterpret_cast<const IType*>(h3.data_ptr()),
                reinterpret_cast<const IType*>(h1_raw.data_ptr()),
                reinterpret_cast<IType*>(dh1_out.data_ptr()),
                reinterpret_cast<IType*>(dh3_out.data_ptr()),
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, H);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 fused silu deriv split2 tiled row bf16 quantize: ", cudaGetErrorString(err));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_split2_row_and_col(
    torch::Tensor input0,
    torch::Tensor input1,
    int mode
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;

    auto device = input0.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty(
        {M, K_total / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M / 128, K_total / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {K_total, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {K_total / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_init;
    ensure_device_init(split2_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);

    switch (mode) {
        case 1:
            mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
        case 2:
            mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
        default:
            mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 row/col quantize: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


void mxfp4_quantize_split2_row_and_col_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_init;
    ensure_device_init(split2_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
        case 2:
            mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
        default:
            mxfp4_v4_split2_rowcol_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 row/col quantize: ", cudaGetErrorString(err));
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
void mxfp4_quantize_split2_row_and_col_opt_launch_inplace_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_opt_init;
    ensure_device_init(split2_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_split2_rowcol_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_split2_rowcol_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in0, tma_in1,
            tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            M, K0, K1, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 row/col quantize opt: ", cudaGetErrorString(err));
}

void mxfp4_quantize_split2_row_and_col_opt_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define SPLIT2_ROWCOL_OPT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_split2_row_and_col_opt_launch_inplace_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG>( \
        input0, input1, row_fp4, row_sc, col_fp4, col_sc, rng_seed, rng_subsequence)
    if (use_rht) {
        if (rht_block_size == 16) {
            if (with_random_sign_mask) {
                if (data_stochastic_rounding && scale_stochastic_rounding) {
                    switch (mode) {
                        case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, true); return;
                        case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, true); return;
                        default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, true); return;
                    }
                }
                if (data_stochastic_rounding) {
                    switch (mode) {
                        case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, true); return;
                        case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, true); return;
                        default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, true); return;
                    }
                }
                if (scale_stochastic_rounding) {
                    switch (mode) {
                        case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, true); return;
                        case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, true); return;
                        default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, true); return;
                    }
                }
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, true); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, true); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, true); return;
                }
            }
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, false); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, false); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, false); return;
                }
            }
            if (data_stochastic_rounding) {
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, false); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, false); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, false); return;
                }
            }
            if (scale_stochastic_rounding) {
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, false); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, false); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, false); return;
                }
            }
            switch (mode) {
                case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, false); return;
                case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, false); return;
                default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, false); return;
            }
        }
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) {
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, true); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, true); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, true); return;
                }
            }
            if (data_stochastic_rounding) {
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, true); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, true); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, true); return;
                }
            }
            if (scale_stochastic_rounding) {
                switch (mode) {
                    case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, true); return;
                    case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, true); return;
                    default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, true); return;
                }
            }
            switch (mode) {
                case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, true); return;
                case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, true); return;
                default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, true); return;
            }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) {
            switch (mode) {
                case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, false); return;
                case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, false); return;
                default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, false); return;
            }
        }
        if (data_stochastic_rounding) {
            switch (mode) {
                case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, false); return;
                case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, false); return;
                default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, false); return;
            }
        }
        if (scale_stochastic_rounding) {
            switch (mode) {
                case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, false); return;
                case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, false); return;
                default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, false); return;
            }
        }
        switch (mode) {
            case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, false); return;
            case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, false); return;
            default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, false); return;
        }
    }

    if (data_stochastic_rounding && scale_stochastic_rounding) {
        switch (mode) {
            case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, false, 16, true); return;
            case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, false, 16, true); return;
            default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, false, 16, true); return;
        }
    }
    if (data_stochastic_rounding) {
        switch (mode) {
            case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, false, 16, true); return;
            case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, false, 16, true); return;
            default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, false, 16, true); return;
        }
    }
    if (scale_stochastic_rounding) {
        switch (mode) {
            case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, false, 16, true); return;
            case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, false, 16, true); return;
            default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, false, 16, true); return;
        }
    }
    switch (mode) {
        case 1: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, false, 16, true); return;
        case 2: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, false, 16, true); return;
        default: SPLIT2_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, false, 16, true); return;
    }
#undef SPLIT2_ROWCOL_OPT_DISPATCH
}

template<QuantMode MODE>
void mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_persistent_resident_init;
    ensure_device_init(split2_persistent_resident_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_split2_rowcol_persistent_resident_kernel_opt<MODE, true, false, false, 16, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    int max_bps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_bps,
        mxfp4_v4_split2_rowcol_persistent_resident_kernel_opt<MODE, true, false, false, 16, true>,
        THREADS,
        dshmem);
    int dev = 0;
    int num_sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);

    const int row_ntk = static_cast<int>(K_total / CHUNK_DIM);
    const int col_ntk = static_cast<int>(M / CHUNK_DIM);
    const int total_tiles = row_ntk * col_ntk;
    int num_persistent = std::min(total_tiles, max_bps * num_sms);
    num_persistent = std::max(num_persistent, 1);

    ensure_work_counter();
    cudaMemsetAsync(g_work_counter, 0, sizeof(unsigned int), stream);
    PersistentArgs args;
    args.work_counter = g_work_counter;
    args.tiles_X = row_ntk;
    args.tiles_Y = col_ntk;
    args.total_tiles = total_tiles;

    mxfp4_v4_split2_rowcol_persistent_resident_kernel_opt<MODE, true, false, false, 16, true>
        <<<num_persistent, THREADS, dshmem, stream>>>(
            tma_in0, tma_in1,
            tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            M, K0, K1, rng_seed, rng_subsequence, args);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 row/col data-SR persistent resident quantize: ", cudaGetErrorString(err));
}

void mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    switch (mode) {
        case 1:
            mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace_impl<QuantMode::ENCODE>(
                input0, input1, row_fp4, row_sc, col_fp4, col_sc, rng_seed, rng_subsequence);
            return;
        case 2:
            mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace_impl<QuantMode::DECODE>(
                input0, input1, row_fp4, row_sc, col_fp4, col_sc, rng_seed, rng_subsequence);
            return;
        default:
            mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace_impl<QuantMode::RTE>(
                input0, input1, row_fp4, row_sc, col_fp4, col_sc, rng_seed, rng_subsequence);
            return;
    }
}


void mxfp4_quantize_split2_row_only_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    int mode
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_row_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_row_init;
    ensure_device_init(split2_row_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split2_row_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_row_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_row_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_split2_row_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, K0, K1);
            break;
        case 2:
            mxfp4_v4_split2_row_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, K0, K1);
            break;
        default:
            mxfp4_v4_split2_row_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                M, K0, K1);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 row-only quantize: ", cudaGetErrorString(err));
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
void mxfp4_quantize_split2_row_only_opt_launch_inplace_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
        TORCH_CHECK(K0 % RHT_BLOCK_SIZE == 0 && K1 % RHT_BLOCK_SIZE == 0,
                    "split widths must be divisible by the RHT block size");
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_row_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_row_opt_init;
    ensure_device_init(split2_row_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_split2_row_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_split2_row_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in0, tma_in1,
            tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            M, K0, K1, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 row-only opt quantize: ", cudaGetErrorString(err));
}

void mxfp4_quantize_split2_row_only_opt_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define SPLIT2_ROW_ONLY_OPT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_split2_row_only_opt_launch_inplace_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG>( \
        input0, input1, row_fp4, row_sc, rng_seed, rng_subsequence)
    if (use_rht) {
        if (rht_block_size == 16) {
            if (with_random_sign_mask) {
                if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, true); return; } }
                if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, true); return; } }
                if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, true); return; } }
                switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, true); return; }
            }
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, false); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, false); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, false); return; } }
            switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, false); return; }
        }
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, true); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, true); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, true); return; } }
            switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, true); return; }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, false); return; } }
        if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, false); return; } }
        if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, false); return; } }
        switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, false); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, false); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, false); return; }
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, false, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, false, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, false, 16, true); return; } }
    if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, false, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, false, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, false, 16, true); return; } }
    if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, false, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, false, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, false, 16, true); return; } }
    switch (mode) { case 1: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, false, 16, true); return; case 2: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, false, 16, true); return; default: SPLIT2_ROW_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, false, 16, true); return; }
#undef SPLIT2_ROW_ONLY_OPT_DISPATCH
}


void mxfp4_quantize_split2_col_only_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_col_init;
    ensure_device_init(split2_col_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split2_col_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_col_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split2_col_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(M / CHUNK_DIM, K_total / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_split2_col_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
        case 2:
            mxfp4_v4_split2_col_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
        default:
            mxfp4_v4_split2_col_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1,
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 col-only quantize: ", cudaGetErrorString(err));
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
void mxfp4_quantize_split2_col_only_opt_launch_inplace_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {input0, input1}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K_total = K0 + K1;
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split2_col_opt_init;
    ensure_device_init(split2_col_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_split2_col_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(M / CHUNK_DIM, K_total / CHUNK_DIM);
    mxfp4_v4_split2_col_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in0, tma_in1,
            tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            M, K0, K1, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split2 col-only opt quantize: ", cudaGetErrorString(err));
}

void mxfp4_quantize_split2_col_only_opt_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define SPLIT2_COL_ONLY_OPT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_split2_col_only_opt_launch_inplace_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG>( \
        input0, input1, col_fp4, col_sc, rng_seed, rng_subsequence)
    if (use_rht) {
        if (rht_block_size == 16) {
            if (with_random_sign_mask) {
                if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, true); return; } }
                if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, true); return; } }
                if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, true); return; } }
                switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, true); return; }
            }
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, false); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, false); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, false); return; } }
            switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, false); return; }
        }
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, true); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, true); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, true); return; } }
            switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, true); return; }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, false); return; } }
        if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, false); return; } }
        if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, false); return; } }
        switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, false); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, false); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, false); return; }
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, true, false, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, true, false, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, true, false, 16, true); return; } }
    if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, true, false, false, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, true, false, false, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, true, false, false, 16, true); return; } }
    if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, true, false, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, true, false, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, true, false, 16, true); return; } }
    switch (mode) { case 1: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::ENCODE, false, false, false, 16, true); return; case 2: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::DECODE, false, false, false, 16, true); return; default: SPLIT2_COL_ONLY_OPT_DISPATCH(QuantMode::RTE, false, false, false, 16, true); return; }
#undef SPLIT2_COL_ONLY_OPT_DISPATCH
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_split3_row_and_col(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    int mode
) {
    for (const auto& input : {input0, input1, input2}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K2 = input2.size(1);
    const int64_t K_total = K0 + K1 + K2;

    auto device = input0.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty(
        {M, K_total / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M / 128, K_total / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {K_total, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {K_total / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_in2{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_in2, input2.data_ptr(), M, K2, TILE_DIM, TILE_DIM, K2, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split3_init;
    ensure_device_init(split3_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);

    switch (mode) {
        case 1:
            mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1, K2);
            break;
        case 2:
            mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1, K2);
            break;
        default:
            mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1, K2);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split3 row/col quantize: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


void mxfp4_quantize_split3_row_and_col_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {input0, input1, input2}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K2 = input2.size(1);
    const int64_t K_total = K0 + K1 + K2;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_in2{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_in2, input2.data_ptr(), M, K2, TILE_DIM, TILE_DIM, K2, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split3_init;
    ensure_device_init(split3_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    switch (mode) {
        case 1:
            mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1, K2);
            break;
        case 2:
            mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1, K2);
            break;
        default:
            mxfp4_v4_split3_rowcol_fused_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                M, K0, K1, K2);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split3 row/col quantize: ", cudaGetErrorString(err));
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
void mxfp4_quantize_split3_row_and_col_opt_launch_inplace_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {input0, input1, input2}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");
    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K2 = input2.size(1);
    const int64_t K_total = K0 + K1 + K2;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_in2{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_in2, input2.data_ptr(), M, K2, TILE_DIM, TILE_DIM, K2, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split3_opt_init;
    ensure_device_init(split3_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_split3_rowcol_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_split3_rowcol_fused_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in0, tma_in1, tma_in2,
            tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            M, K0, K1, K2, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 split3 row/col quantize opt: ", cudaGetErrorString(err));
}

void mxfp4_quantize_split3_row_and_col_opt_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define SPLIT3_ROWCOL_OPT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_split3_row_and_col_opt_launch_inplace_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG>( \
        input0, input1, input2, row_fp4, row_sc, col_fp4, col_sc, rng_seed, rng_subsequence)
    if (use_rht) {
        if (rht_block_size == 16) {
            if (with_random_sign_mask) {
                if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, true); return; } }
                if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, true); return; } }
                if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, true); return; } }
                switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, true); return; }
            }
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, false); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, false); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, false); return; } }
            switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, false); return; }
        }
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, true); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, true); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, true); return; } }
            switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, true); return; }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, false); return; } }
        if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, false); return; } }
        if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, false); return; } }
        switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, false); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, false); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, false); return; }
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, false, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, false, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, false, 16, true); return; } }
    if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, false, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, false, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, false, 16, true); return; } }
    if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, false, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, false, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, false, 16, true); return; } }
    switch (mode) { case 1: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, false, 16, true); return; case 2: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, false, 16, true); return; default: SPLIT3_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, false, 16, true); return; }
#undef SPLIT3_ROWCOL_OPT_DISPATCH
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_quantize_split3_row_and_col_inverse_rope_live64(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    int mode
) {
    for (const auto& input : {input0, input1, input2}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");
    check_rope_live64_tensor(rope_cs, rope_seq_len);

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K2 = input2.size(1);
    const int64_t K_total = K0 + K1 + K2;

    auto device = input0.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty(
        {M, K_total / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty(
        {M / 128, K_total / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty(
        {K_total, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty(
        {K_total / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_in2{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_in2, input2.data_ptr(), M, K2, TILE_DIM, TILE_DIM, K2, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split3_inverse_rope_init;
    ensure_device_init(split3_inverse_rope_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    const int seq_mask = static_cast<int>(rope_seq_len - 1);
    const float2* rope_ptr = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>());

    switch (mode) {
        case 1:
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                rope_ptr, seq_mask, M, K0, K1, K2);
            break;
        case 2:
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                rope_ptr, seq_mask, M, K0, K1, K2);
            break;
        default:
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                rope_ptr, seq_mask, M, K0, K1, K2);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "mxfp4 split3 row/col inverse rope live64 quantize: ",
                cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}


void mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode
) {
    for (const auto& input : {input0, input1, input2}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");
    check_rope_live64_tensor(rope_cs, rope_seq_len);

    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K2 = input2.size(1);
    const int64_t K_total = K0 + K1 + K2;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));

    auto stream = at::cuda::getCurrentCUDAStream();

    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_in2{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_in2, input2.data_ptr(), M, K2, TILE_DIM, TILE_DIM, K2, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split3_inverse_rope_init;
    ensure_device_init(split3_inverse_rope_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    const int seq_mask = static_cast<int>(rope_seq_len - 1);
    const float2* rope_ptr = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>());
    switch (mode) {
        case 1:
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::ENCODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                rope_ptr, seq_mask, M, K0, K1, K2);
            break;
        case 2:
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::DECODE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                rope_ptr, seq_mask, M, K0, K1, K2);
            break;
        default:
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel<QuantMode::RTE><<<grid, THREADS, dshmem, stream>>>(
                tma_in0, tma_in1, tma_in2,
                tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
                tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
                rope_ptr, seq_mask, M, K0, K1, K2);
            break;
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "mxfp4 split3 row/col inverse rope live64 quantize: ",
                cudaGetErrorString(err));
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
void mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace_impl(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    for (const auto& input : {input0, input1, input2}) {
        TORCH_CHECK(input.is_cuda() && input.is_contiguous());
        TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
        TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0,
                    "split inputs must be contiguous bf16 matrices with dims divisible by 128");
    }
    TORCH_CHECK(input0.size(0) == input1.size(0) && input0.size(0) == input2.size(0),
                "split inputs must have the same M dimension");
    check_rope_live64_tensor(rope_cs, rope_seq_len);
    const int64_t M = input0.size(0);
    const int64_t K0 = input0.size(1);
    const int64_t K1 = input1.size(1);
    const int64_t K2 = input2.size(1);
    const int64_t K_total = K0 + K1 + K2;
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous());
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({M, K_total / 2}));
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous());
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({M / 128, K_total / 128, 32, 16}));
    TORCH_CHECK(col_fp4.is_cuda() && col_fp4.is_contiguous());
    TORCH_CHECK(col_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2);
    TORCH_CHECK(col_fp4.sizes() == torch::IntArrayRef({K_total, M / 2}));
    TORCH_CHECK(col_sc.is_cuda() && col_sc.is_contiguous());
    TORCH_CHECK(col_sc.scalar_type() == torch::kUInt8);
    TORCH_CHECK(col_sc.sizes() == torch::IntArrayRef({K_total / 128, M / 128, 32, 16}));
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    alignas(64) CUtensorMap tma_in0{}, tma_in1{}, tma_in2{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in0, input0.data_ptr(), M, K0, TILE_DIM, TILE_DIM, K0, 16);
    create_tma_2d(tma_in1, input1.data_ptr(), M, K1, TILE_DIM, TILE_DIM, K1, 16);
    create_tma_2d(tma_in2, input2.data_ptr(), M, K2, TILE_DIM, TILE_DIM, K2, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K_total, TILE_DIM, TILE_DIM, K_total, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K_total, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_rowcol_shmem_size();
    static DeviceInitFlags split3_inverse_rope_opt_init;
    ensure_device_init(split3_inverse_rope_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(K_total / CHUNK_DIM, M / CHUNK_DIM);
    const int seq_mask = static_cast<int>(rope_seq_len - 1);
    const float2* rope_ptr = reinterpret_cast<const float2*>(rope_cs.data_ptr<float>());
    mxfp4_v4_split3_rowcol_inverse_rope_live64_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in0, tma_in1, tma_in2,
            tma_row_out, reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            tma_col_out, reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            rope_ptr, seq_mask, M, K0, K1, K2, rng_seed, rng_subsequence);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "mxfp4 split3 row/col inverse rope live64 quantize opt: ",
                cudaGetErrorString(err));
}

void mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace(
    torch::Tensor input0,
    torch::Tensor input1,
    torch::Tensor input2,
    torch::Tensor rope_cs,
    int64_t rope_seq_len,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    torch::Tensor col_fp4,
    torch::Tensor col_sc,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define SPLIT3_ROPE_OPT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG) \
    mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG>( \
        input0, input1, input2, rope_cs, rope_seq_len, row_fp4, row_sc, col_fp4, col_sc, rng_seed, rng_subsequence)
    if (use_rht) {
        if (rht_block_size == 16) {
            if (with_random_sign_mask) {
                if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, true); return; } }
                if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, true); return; } }
                if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, true); return; } }
                switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, true); return; }
            }
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, false); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, false); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, false); return; } }
            switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, false); return; }
        }
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, true); return; } }
            if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, true); return; } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, true); return; } }
            switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, true); return; }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, false); return; } }
        if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, false); return; } }
        if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, false); return; } }
        switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, false); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, false); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, false); return; }
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, true, false, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, true, false, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, true, false, 16, true); return; } }
    if (data_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, true, false, false, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, true, false, false, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, true, false, false, 16, true); return; } }
    if (scale_stochastic_rounding) { switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, true, false, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, true, false, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, true, false, 16, true); return; } }
    switch (mode) { case 1: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::ENCODE, false, false, false, 16, true); return; case 2: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::DECODE, false, false, false, 16, true); return; default: SPLIT3_ROPE_OPT_DISPATCH(QuantMode::RTE, false, false, false, 16, true); return; }
#undef SPLIT3_ROPE_OPT_DISPATCH
}



template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col_impl(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16 && norm_weight.dim() == 1);

    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(norm_weight.size(0) == K, "norm_weight must have shape (K,)");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto inv_rms = torch::empty({M},
        torch::dtype(torch::kFloat32).device(device));

    compute_inv_rms_kernel<RMS_BLOCK_THREADS><<<M, RMS_BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const IType*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        epsilon,
        static_cast<int>(M),
        static_cast<int>(K));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 fused rmsnorm inv_rms: ", cudaGetErrorString(err));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{}, tma_norm_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_fused_rmsnorm_shmem_size();
    static DeviceInitFlags fused_rms_init;
    ensure_device_init(fused_rms_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_fused_rmsnorm_rowcol_kernel<MODE><<<grid, THREADS, dshmem, stream>>>(
        tma_in,
        tma_row_out,
        tma_col_out,
        tma_norm_out,
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<const IType*>(norm_weight.data_ptr()),
        M,
        K,
        false);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 fused rmsnorm rowcol quantize: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon,
    int mode
) {
    switch (mode) {
        case 1:
            return mxfp4_fused_rmsnorm_quantize_row_and_col_impl<QuantMode::ENCODE>(
                input, norm_weight, epsilon);
        case 2:
            return mxfp4_fused_rmsnorm_quantize_row_and_col_impl<QuantMode::DECODE>(
                input, norm_weight, epsilon);
        default:
            return mxfp4_fused_rmsnorm_quantize_row_and_col_impl<QuantMode::RTE>(
                input, norm_weight, epsilon);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor row_rms_partial,
    float epsilon,
    int mode
) {
    TORCH_CHECK(mode == 1,
                "MXFP4 exact C/D/E production path requires encode mode");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                    input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be contiguous CUDA bf16 [M,K]");
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous() &&
                    norm_weight.scalar_type() == torch::kBFloat16 &&
                    norm_weight.dim() == 1,
                "norm_weight must be contiguous CUDA bf16 [K]");
    TORCH_CHECK(row_rms_partial.is_cuda() && row_rms_partial.is_contiguous() &&
                    row_rms_partial.scalar_type() == torch::kFloat32 &&
                    row_rms_partial.dim() == 2,
                "row_rms_partial must be contiguous CUDA fp32 [M,K/256]");
    TORCH_CHECK(input.device() == norm_weight.device() &&
                    input.device() == row_rms_partial.device(),
                "input, norm_weight, and row_rms_partial must share one CUDA device");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(M > 0 && M % 256 == 0 && K == 4096,
                "MXFP4 exact C/D/E consumer requires [M,4096] with M divisible by 256");
    TORCH_CHECK(norm_weight.size(0) == K,
                "norm_weight must match input K");
    TORCH_CHECK(row_rms_partial.size(0) == M &&
                    row_rms_partial.size(1) == K / 256,
                "row_rms_partial shape must be [M,K/256]");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0.0f,
                "epsilon must be finite and non-negative");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto inv_rms = torch::empty({M},
        torch::dtype(torch::kFloat32).device(device));

    c1_rms_reduce::row_rms_reduce_entrypoint(
        row_rms_partial, inv_rms, K, epsilon);

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{}, tma_norm_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_fused_rmsnorm_shmem_size();
    static DeviceInitFlags fused_rms_partial_init;
    ensure_device_init(fused_rms_partial_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::ENCODE>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in,
            tma_row_out,
            tma_col_out,
            tma_norm_out,
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            inv_rms.data_ptr<float>(),
            reinterpret_cast<const IType*>(norm_weight.data_ptr()),
            M,
            K,
            false);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "MXFP4 exact C/D/E row/col quantize: ",
                cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, inv_rms);
}

template<QuantMode MODE>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col_with_output_impl(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16 && norm_weight.dim() == 1);

    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(norm_weight.size(0) == K, "norm_weight must have shape (K,)");

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto normed = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto inv_rms = torch::empty({M},
        torch::dtype(torch::kFloat32).device(device));

    compute_inv_rms_kernel<RMS_BLOCK_THREADS><<<M, RMS_BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const IType*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        epsilon,
        static_cast<int>(M),
        static_cast<int>(K));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 fused rmsnorm inv_rms with output: ", cudaGetErrorString(err));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{}, tma_norm_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);
    create_tma_2d(tma_norm_out, normed.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);

    const int dshmem = v4_fused_rmsnorm_shmem_size();
    static DeviceInitFlags fused_rms_init;
    ensure_device_init(fused_rms_init, [&] {
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::RTE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::ENCODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
        cudaFuncSetAttribute(mxfp4_v4_fused_rmsnorm_rowcol_kernel<QuantMode::DECODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_fused_rmsnorm_rowcol_kernel<MODE><<<grid, THREADS, dshmem, stream>>>(
        tma_in,
        tma_row_out,
        tma_col_out,
        tma_norm_out,
        reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
        reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
        inv_rms.data_ptr<float>(),
        reinterpret_cast<const IType*>(norm_weight.data_ptr()),
        M,
        K,
        true);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 fused rmsnorm rowcol quantize with output: ", cudaGetErrorString(err));

    return std::make_tuple(normed, row_fp4, row_sc, col_fp4, col_sc, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col_with_output(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon,
    int mode
) {
    switch (mode) {
        case 1:
            return mxfp4_fused_rmsnorm_quantize_row_and_col_with_output_impl<QuantMode::ENCODE>(
                input, norm_weight, epsilon);
        case 2:
            return mxfp4_fused_rmsnorm_quantize_row_and_col_with_output_impl<QuantMode::DECODE>(
                input, norm_weight, epsilon);
        default:
            return mxfp4_fused_rmsnorm_quantize_row_and_col_with_output_impl<QuantMode::RTE>(
                input, norm_weight, epsilon);
    }
}

template<QuantMode MODE, bool DATA_SR, bool SCALE_SR, bool WITH_RHT = false, int RHT_BLOCK_SIZE = 16, bool WITH_RANDOM_SIGN_MASK = true>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col_opt_impl(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(norm_weight.is_cuda() && norm_weight.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16 && norm_weight.dim() == 1);

    int64_t M = input.size(0), K = input.size(1);
    TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "M and K must be multiples of 128");
    TORCH_CHECK(norm_weight.size(0) == K, "norm_weight must have shape (K,)");
    if constexpr (WITH_RHT) {
        TORCH_CHECK(RHT_BLOCK_SIZE == 16 || RHT_BLOCK_SIZE == 32, "RHT block size must be 16 or 32");
        TORCH_CHECK(M % RHT_BLOCK_SIZE == 0, "M must be divisible by the RHT block size");
    }

    auto device = input.device();
    auto stream = at::cuda::getCurrentCUDAStream();

    auto row_fp4 = torch::empty({M, K / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({M / 128, K / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({K, M / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({K / 128, M / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto inv_rms = torch::empty({M},
        torch::dtype(torch::kFloat32).device(device));

    compute_inv_rms_kernel<RMS_BLOCK_THREADS><<<M, RMS_BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const IType*>(input.data_ptr()),
        inv_rms.data_ptr<float>(),
        epsilon,
        static_cast<int>(M),
        static_cast<int>(K));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 fused rmsnorm inv_rms opt: ", cudaGetErrorString(err));

    alignas(64) CUtensorMap tma_in{}, tma_row_out{}, tma_col_out{};
    create_tma_2d(tma_in, input.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 16);
    create_tma_2d(tma_row_out, row_fp4.data_ptr(), M, K, TILE_DIM, TILE_DIM, K, 4);
    create_tma_2d(tma_col_out, col_fp4.data_ptr(), K, M, TILE_DIM, TILE_DIM, M, 4);

    const int dshmem = v4_fused_rmsnorm_shmem_size();
    static DeviceInitFlags fused_rmsnorm_opt_init;
    ensure_device_init(fused_rmsnorm_opt_init, [&] {
        cudaFuncSetAttribute(
            mxfp4_v4_fused_rmsnorm_rowcol_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dshmem);
    });

    dim3 grid(K / CHUNK_DIM, M / CHUNK_DIM);
    mxfp4_v4_fused_rmsnorm_rowcol_kernel_opt<MODE, DATA_SR, SCALE_SR, WITH_RHT, RHT_BLOCK_SIZE, WITH_RANDOM_SIGN_MASK>
        <<<grid, THREADS, dshmem, stream>>>(
            tma_in,
            tma_row_out,
            tma_col_out,
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            inv_rms.data_ptr<float>(),
            reinterpret_cast<const IType*>(norm_weight.data_ptr()),
            M,
            K,
            rng_seed,
            rng_subsequence);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mxfp4 v4 fused rmsnorm rowcol quantize opt: ", cudaGetErrorString(err));

    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc, inv_rms);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_fused_rmsnorm_quantize_row_and_col_opt(
    torch::Tensor input,
    torch::Tensor norm_weight,
    float epsilon,
    int mode,
    bool data_stochastic_rounding,
    bool scale_stochastic_rounding,
    bool use_rht,
    int rht_block_size,
    bool with_random_sign_mask,
    uint64_t rng_seed,
    uint64_t rng_subsequence
) {
#define FUSED_RMS_ROWCOL_OPT_DISPATCH(MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG) \
    return mxfp4_fused_rmsnorm_quantize_row_and_col_opt_impl<MODE_ENUM, DATA_SR_FLAG, SCALE_SR_FLAG, WITH_RHT_FLAG, RHT_BLOCK, SIGN_FLAG>( \
        input, norm_weight, epsilon, rng_seed, rng_subsequence)
    if (use_rht) {
        if (rht_block_size == 16) {
            if (with_random_sign_mask) {
                if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, true); } }
                if (data_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, true); } }
                if (scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, true); } }
                switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, true); }
            }
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 16, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 16, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 16, false); } }
            if (data_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 16, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 16, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 16, false); } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 16, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 16, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 16, false); } }
            switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 16, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 16, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 16, false); }
        }
        if (with_random_sign_mask) {
            if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, true); } }
            if (data_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, true); } }
            if (scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, true); } }
            switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, true); }
        }
        if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, true, 32, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, true, 32, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, true, 32, false); } }
        if (data_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, true, 32, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, true, 32, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, true, 32, false); } }
        if (scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, true, 32, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, true, 32, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, true, 32, false); } }
        switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, true, 32, false); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, true, 32, false); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, true, 32, false); }
    }
    if (data_stochastic_rounding && scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, true, false, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, true, false, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, true, false, 16, true); } }
    if (data_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, true, false, false, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, true, false, false, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, true, false, false, 16, true); } }
    if (scale_stochastic_rounding) { switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, true, false, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, true, false, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, true, false, 16, true); } }
    switch (mode) { case 1: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::ENCODE, false, false, false, 16, true); case 2: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::DECODE, false, false, false, 16, true); default: FUSED_RMS_ROWCOL_OPT_DISPATCH(QuantMode::RTE, false, false, false, 16, true); }
#undef FUSED_RMS_ROWCOL_OPT_DISPATCH
}


// ═══════════════════════════════════════════════════════════════════
// PyBind11
// ═══════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_quantize_for_gemm", &mxfp4_quantize_for_gemm,
          "MXFP4 v3 quantize (pipelined) → (fp4, scales). mode: 0=RTE, 1=ENCODE, 2=DECODE",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_for_gemm_opt", &mxfp4_quantize_for_gemm_opt,
          "MXFP4 v3 quantize with optional data/scale stochastic rounding",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rng_seed") = 1234,
          py::arg("rng_subsequence") = 0);
    m.def("mxfp4_quantize_for_gemm_opt_rht", &mxfp4_quantize_for_gemm_opt_rht,
          "MXFP4 v3 quantize with fused register/shared-memory butterfly RHT and optional SR",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234,
          py::arg("rng_subsequence") = 0);
    m.def("mxfp4_group_quantize_dim0", &mxfp4_group_quantize_dim0,
          "MXFP4 v4 group quantize contiguous input → list[(fp4, scales)]");
    m.def("mxfp4_group_quantize_dim0_opt", &mxfp4_group_quantize_dim0_opt,
          "MXFP4 v4 grouped quantize with optional fused RHT and stochastic rounding",
          py::arg("input"), py::arg("group_sizes"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234,
          py::arg("rng_subsequence") = 0);
    m.def("mxfp4_multi_quantize", &mxfp4_multi_quantize,
          "MXFP4 v4 multi-tensor quantize → list[(fp4, scales)]");
    m.def("mxfp4_multi_quantize_opt", &mxfp4_multi_quantize_opt,
          "MXFP4 v4 multi-tensor quantize with optional fused RHT and stochastic rounding",
          py::arg("inputs"),
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234,
          py::arg("rng_subsequence") = 0);
    m.def("mxfp4_quantize_col_only", &mxfp4_quantize_col_only,
          "MXFP4 v4 quantize transpose contract directly from BF16 input → (fp4, scales)",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_col_only_opt", &mxfp4_quantize_col_only_opt,
          "MXFP4 v4 col-only quantize with optional data/scale stochastic rounding",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_col_only_opt_rht", &mxfp4_quantize_col_only_opt_rht,
          "MXFP4 v4 col-contract quantize with fused M-axis butterfly RHT and optional stochastic rounding",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_row_and_col", &mxfp4_quantize_row_and_col,
          "MXFP4 v4 quantize row + direct col quantize → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_weight_2d", &mxfp4_quantize_weight_2d,
          "MXFP4 v4 shared 32x32 weight quantize with exact transpose contract",
          py::arg("input"));
    m.def("mxfp4_quantize_nhsd_wo_row_and_col", &mxfp4_quantize_nhsd_wo_row_and_col,
          "MXFP4 v4 quantize contiguous [B,H,S,D] attention output as logical WO [B*S,H*D]",
          py::arg("input"), py::arg("mode") = 0);
    m.def("mxfp4_pack_grouped_rows_bf16", &mxfp4_pack_grouped_rows_bf16,
          "Pack BF16 grouped rows with padded row slots in one CUDA launch",
          py::arg("input"), py::arg("starts"), py::arg("rows"), py::arg("padded_rows"), py::arg("output_cols"));
    m.def("mxfp4_pack_indexed_rows_bf16", &mxfp4_pack_indexed_rows_bf16,
          "Gather indexed BF16 rows directly into uniform padded grouped row slots",
          py::arg("input"), py::arg("token_indices"),
          py::arg("num_batches"), py::arg("live_rows_per_batch"),
          py::arg("padded_rows_per_batch"), py::arg("output_cols"));
    m.def("mxfp4_pack_grouped_rows_quantize_row_and_col", &mxfp4_pack_grouped_rows_quantize_row_and_col,
          "Pack uniform grouped BF16 rows and quantize row+col contracts in one CUDA launch",
          py::arg("input"), py::arg("num_batches"), py::arg("live_rows_per_batch"),
          py::arg("padded_rows_per_batch"), py::arg("output_cols"), py::arg("mode") = 1);
    m.def("mxfp4_pack_indexed_scaled_rows_quantize_row_and_col", &mxfp4_pack_indexed_scaled_rows_quantize_row_and_col,
          "Gather indexed BF16 rows, scale by FP32 routing scores, and quantize row+col contracts",
          py::arg("input"), py::arg("token_indices"), py::arg("scores"),
          py::arg("num_batches"), py::arg("live_rows_per_batch"),
          py::arg("padded_rows_per_batch"), py::arg("output_cols"), py::arg("mode") = 1);
    m.def("mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col", &mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col,
          "Gather indexed raw BF16 rows, apply RMSNorm from inv_rms/weight, and quantize row+col contracts",
          py::arg("input"), py::arg("norm_weight"), py::arg("inv_rms"), py::arg("token_indices"),
          py::arg("num_batches"), py::arg("live_rows_per_batch"),
          py::arg("padded_rows_per_batch"), py::arg("output_cols"), py::arg("mode") = 1);
    m.def("mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable", &mxfp4_pack_indexed_scaled_rows_quantize_row_and_col_variable,
          "Gather variable-count indexed BF16 rows, scale by FP32 routing scores, and quantize row+col contracts",
          py::arg("input"), py::arg("token_indices"), py::arg("scores"),
          py::arg("route_starts"), py::arg("rows"), py::arg("padded_starts"), py::arg("padded_rows"),
          py::arg("output_cols"), py::arg("mode") = 1);
    m.def("mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable", &mxfp4_pack_indexed_rmsnorm_rows_quantize_row_and_col_variable,
          "Gather variable-count indexed raw BF16 rows, apply RMSNorm from inv_rms/weight, and quantize row+col contracts",
          py::arg("input"), py::arg("norm_weight"), py::arg("inv_rms"), py::arg("token_indices"),
          py::arg("route_starts"), py::arg("rows"), py::arg("padded_starts"), py::arg("padded_rows"),
          py::arg("output_cols"), py::arg("mode") = 1);
    m.def("mxfp4_pack_w13_bf16", &mxfp4_pack_w13_bf16,
          "Pack grouped expert w1/w3 weights into [E*H13n, Dk] without torch.cat",
          py::arg("w1"), py::arg("w3"), py::arg("H13n"), py::arg("Dk"));
    m.def("mxfp4_split_w13_bf16", &mxfp4_split_w13_bf16,
          "Split combined grouped expert w13 gradient into contiguous w1/w3 gradients",
          py::arg("grad_w13"), py::arg("H"), py::arg("D"));
    m.def("mxfp4_scatter_grouped_rows_bf16", &mxfp4_scatter_grouped_rows_bf16,
          "Scatter BF16 packed grouped rows back to live row slots in one CUDA launch",
          py::arg("input"), py::arg("output"), py::arg("starts"), py::arg("rows"), py::arg("padded_rows"));
    m.def("mxfp4_copy_col_slices", &mxfp4_copy_col_slices,
          "Copy contiguous column-format MXFP4 slices from one bulk row+col quantized tensor",
          py::arg("col_fp4"), py::arg("col_sc"), py::arg("row_starts"), py::arg("rows"));
    m.def("mxfp4_quantize_row_and_col_opt", &mxfp4_quantize_row_and_col_opt,
          "MXFP4 v4 row+col quantize with optional data/scale stochastic rounding",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_row_and_col_opt_rht", &mxfp4_quantize_row_and_col_opt_rht,
          "MXFP4 v4 row+col quantize with fused M-axis butterfly RHT on the col contract",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_row_and_col_opt_rht_both", &mxfp4_quantize_row_and_col_opt_rht_both,
          "MXFP4 v4 row+col quantize with K-axis row-contract RHT and M-axis col-contract RHT",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_row_and_col_opt_rht_row_only", &mxfp4_quantize_row_and_col_opt_rht_row_only,
          "MXFP4 v4 row+col quantize with fused K-axis RHT only on the row-output contract",
          py::arg("input"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_row_and_col_launch_inplace", &mxfp4_quantize_row_and_col_launch_inplace,
          "MXFP4 v4 quantize row + direct col quantize into preallocated outputs",
          py::arg("input"), py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col", &mxfp4_fused_silu_mul_quantize_row_and_col,
          "MXFP4 fused silu mul + row/col quantize → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("h1_raw"), py::arg("h3"), py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace",
          &mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace,
          "MXFP4 fused silu mul + row/col quantize into preallocated outputs",
          py::arg("h1_raw"), py::arg("h3"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace",
          &mxfp4_fused_silu_mul_sigmoid_quantize_row_and_col_launch_inplace,
          "MXFP4 fused silu mul + saved sigmoid + row/col quantize into preallocated outputs",
          py::arg("h1_raw"), py::arg("h3"), py::arg("sig_h1"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_sqrelu_quantize_row_and_col", &mxfp4_fused_sqrelu_quantize_row_and_col,
          "MXFP4 fused square-ReLU + row/col quantize -> (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("h1_raw"), py::arg("mode") = 0);
    m.def("mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace",
          &mxfp4_fused_sqrelu_quantize_row_and_col_launch_inplace,
          "MXFP4 fused square-ReLU + row/col quantize into preallocated outputs",
          py::arg("h1_raw"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_sqrelu_deriv_quantize_row_and_col", &mxfp4_fused_sqrelu_deriv_quantize_row_and_col,
          "MXFP4 fused square-ReLU derivative + row/col quantize -> (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("dh"), py::arg("h1_raw"), py::arg("mode") = 0);
    m.def("mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace",
          &mxfp4_fused_sqrelu_deriv_quantize_row_and_col_launch_inplace,
          "MXFP4 fused square-ReLU derivative + row/col quantize into preallocated outputs",
          py::arg("dh"), py::arg("h1_raw"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace",
          &mxfp4_fused_sqrelu_quantize_row_and_col_opt_launch_inplace,
          "MXFP4 fused square-ReLU + row/col quantize with optional RHT/SR into preallocated outputs",
          py::arg("h1_raw"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL,
          py::arg("row_with_rht") = false);
    m.def("mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace",
          &mxfp4_fused_sqrelu_deriv_quantize_row_and_col_opt_launch_inplace,
          "MXFP4 fused square-ReLU derivative + row/col quantize with optional RHT/SR into preallocated outputs",
          py::arg("dh"), py::arg("h1_raw"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL,
          py::arg("row_with_rht") = false);
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only",
          &mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only,
          "MXFP4 fused silu mul + row-RHT row/col quantize",
          py::arg("h1_raw"), py::arg("h3"), py::arg("mode") = 0,
          py::arg("rht_block_size") = 32, py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 1234, py::arg("rng_subsequence") = 0);
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only_launch_inplace",
          &mxfp4_fused_silu_mul_quantize_row_and_col_rht_row_only_launch_inplace,
          "MXFP4 fused silu mul + row-RHT row/col quantize into preallocated outputs",
          py::arg("h1_raw"), py::arg("h3"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0, py::arg("rht_block_size") = 32,
          py::arg("with_random_sign_mask") = false,
          py::arg("rng_seed") = 1234, py::arg("rng_subsequence") = 0);
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace",
          &mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace,
          "MXFP4 fused silu mul + row/col quantize with optional RHT/SR into preallocated outputs",
          py::arg("h1_raw"), py::arg("h3"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL,
          py::arg("row_with_rht") = false);
    m.def("mxfp4_fused_rmsnorm_to_bf16", &mxfp4_fused_rmsnorm_to_bf16,
          "MXFP4 fused RMSNorm to BF16 → (normed_bf16, inv_rms)",
          py::arg("input"), py::arg("norm_weight"), py::arg("epsilon"));
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col_strided",
          &mxfp4_fused_silu_mul_quantize_row_and_col_strided,
          "MXFP4 fused strided packed w1|w3 SiLU*mul row/col quantize → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("h13"), py::arg("H"), py::arg("h3_offset"), py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_mul_quantize_row_and_col_strided_launch_inplace",
          &mxfp4_fused_silu_mul_quantize_row_and_col_strided_launch_inplace,
          "MXFP4 fused strided packed w1|w3 SiLU*mul row/col quantize into preallocated outputs",
          py::arg("h13"), py::arg("H"), py::arg("h3_offset"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_and_col", &mxfp4_fused_silu_deriv_quantize_split2_row_and_col,
          "MXFP4 fused silu-deriv + split2 row/col quantize → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"), py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace",
          &mxfp4_fused_silu_deriv_quantize_split2_row_and_col_launch_inplace,
          "MXFP4 fused silu-deriv + split2 row/col quantize into preallocated outputs",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols",
          &mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols,
          "MXFP4 fused silu-deriv + split2 quantize with stacked row output and split col outputs",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"), py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace",
          &mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace,
          "MXFP4 fused silu-deriv + split2 quantize into stacked row output and split col outputs",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("col0_fp4"), py::arg("col0_sc"),
          py::arg("col1_fp4"), py::arg("col1_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace",
          &mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_and_col_splitcols_launch_inplace,
          "MXFP4 fused saved-sigmoid silu-deriv + split2 quantize into stacked row output and split col outputs",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"), py::arg("sig_h1"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("col0_fp4"), py::arg("col0_sc"),
          py::arg("col1_fp4"), py::arg("col1_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined",
          &mxfp4_fused_silu_deriv_quantize_split2_row_and_col_strided_combined,
          "MXFP4 fused strided silu-deriv + combined split2 row/col quantize for grouped GEMM",
          py::arg("dh"), py::arg("h13"), py::arg("H"), py::arg("h3_offset"), py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace",
          &mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace,
          "MXFP4 fused silu-deriv + split2 row quantize into preallocated outputs plus BF16 branches",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1_out"), py::arg("dh3_out"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace",
          &mxfp4_fused_silu_deriv_from_sigmoid_quantize_split2_row_bf16_launch_inplace,
          "MXFP4 fused saved-sigmoid silu-deriv + split2 row quantize into preallocated outputs plus BF16 branches",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"), py::arg("sig_h1"),
          py::arg("dh1_out"), py::arg("dh3_out"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace",
          &mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace,
          "MXFP4 tiled fused silu-deriv + split2 row quantize into preallocated outputs plus BF16 branches",
          py::arg("dh"), py::arg("h3"), py::arg("h1_raw"),
          py::arg("dh1_out"), py::arg("dh3_out"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_quantize_split2_row_and_col", &mxfp4_quantize_split2_row_and_col,
          "MXFP4 split2 dim1 quantize without BF16 concat → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("input0"), py::arg("input1"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_split2_row_and_col_launch_inplace", &mxfp4_quantize_split2_row_and_col_launch_inplace,
          "MXFP4 split2 dim1 quantize into preallocated outputs",
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_quantize_split2_row_and_col_opt_launch_inplace", &mxfp4_quantize_split2_row_and_col_opt_launch_inplace,
          "MXFP4 split2 dim1 quantize into preallocated outputs with optional fused RHT and stochastic rounding",
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace",
          &mxfp4_quantize_split2_row_and_col_datasr_persistent_launch_inplace,
          "MXFP4 split2 dim1 row+col DATA_SR quantize with persistent resident CTAs",
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_split2_row_only_launch_inplace", &mxfp4_quantize_split2_row_only_launch_inplace,
          "MXFP4 split2 dim1 row-only quantize into preallocated outputs",
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_quantize_split2_row_only_opt_launch_inplace",
          &mxfp4_quantize_split2_row_only_opt_launch_inplace,
          "MXFP4 split2 dim1 row-only quantize into preallocated outputs with optional RHT and stochastic rounding",
          py::arg("input0"), py::arg("input1"),
          py::arg("row_fp4"), py::arg("row_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_split2_col_only_launch_inplace", &mxfp4_quantize_split2_col_only_launch_inplace,
          "MXFP4 split2 dim1 col-only quantize into preallocated outputs",
          py::arg("input0"), py::arg("input1"),
          py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_quantize_split2_col_only_opt_launch_inplace",
          &mxfp4_quantize_split2_col_only_opt_launch_inplace,
          "MXFP4 split2 dim1 col-only quantize into preallocated outputs with optional RHT and stochastic rounding",
          py::arg("input0"), py::arg("input1"),
          py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_split3_row_and_col", &mxfp4_quantize_split3_row_and_col,
          "MXFP4 split3 dim1 quantize without BF16 concat → (row_fp4, row_sc, col_fp4, col_sc)",
          py::arg("input0"), py::arg("input1"), py::arg("input2"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_split3_row_and_col_launch_inplace", &mxfp4_quantize_split3_row_and_col_launch_inplace,
          "MXFP4 split3 dim1 quantize into preallocated outputs",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_quantize_split3_row_and_col_opt_launch_inplace", &mxfp4_quantize_split3_row_and_col_opt_launch_inplace,
          "MXFP4 split3 dim1 quantize into preallocated outputs with optional fused RHT and stochastic rounding",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_quantize_split3_row_and_col_inverse_rope_live64",
          &mxfp4_quantize_split3_row_and_col_inverse_rope_live64,
          "MXFP4 split3 dim1 quantize with fused inverse live64 RoPE on inputs 0/1",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("rope_cs"), py::arg("rope_seq_len"), py::arg("mode") = 0);
    m.def("mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace",
          &mxfp4_quantize_split3_row_and_col_inverse_rope_live64_launch_inplace,
          "MXFP4 split3 dim1 quantize into preallocated outputs with fused inverse live64 RoPE on inputs 0/1",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("rope_cs"), py::arg("rope_seq_len"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0);
    m.def("mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace",
          &mxfp4_quantize_split3_row_and_col_inverse_rope_live64_opt_launch_inplace,
          "MXFP4 split3 dim1 quantize into preallocated outputs with fused inverse live64 RoPE, RHT and stochastic rounding",
          py::arg("input0"), py::arg("input1"), py::arg("input2"),
          py::arg("rope_cs"), py::arg("rope_seq_len"),
          py::arg("row_fp4"), py::arg("row_sc"), py::arg("col_fp4"), py::arg("col_sc"),
          py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
    m.def("mxfp4_fused_rmsnorm_quantize_row_and_col", &mxfp4_fused_rmsnorm_quantize_row_and_col,
          "MXFP4 v4 fused RMSNorm + row/col quantize → (row_fp4, row_sc, col_fp4, col_sc, inv_rms)",
          py::arg("input"), py::arg("norm_weight"), py::arg("epsilon"), py::arg("mode") = 0);
    m.def("mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial",
          &mxfp4_fused_rmsnorm_quantize_row_and_col_from_row_rms_partial,
          "MXFP4 exact RMS partial reduction + fused row/col quantize",
          py::arg("input"), py::arg("norm_weight"),
          py::arg("row_rms_partial"), py::arg("epsilon"),
          py::arg("mode") = 1);
    m.def("mxfp4_fused_rmsnorm_quantize_row_and_col_with_output",
          &mxfp4_fused_rmsnorm_quantize_row_and_col_with_output,
          "MXFP4 v4 fused RMSNorm + BF16 output + row/col quantize → (normed, row_fp4, row_sc, col_fp4, col_sc, inv_rms)",
          py::arg("input"), py::arg("norm_weight"), py::arg("epsilon"), py::arg("mode") = 0);
    m.def("mxfp4_fused_rmsnorm_quantize_row_and_col_opt", &mxfp4_fused_rmsnorm_quantize_row_and_col_opt,
          "MXFP4 fused RMSNorm + row/col quantize with fused RHT and stochastic rounding",
          py::arg("input"), py::arg("norm_weight"), py::arg("epsilon"), py::arg("mode") = 0,
          py::arg("data_stochastic_rounding") = false,
          py::arg("scale_stochastic_rounding") = false,
          py::arg("use_rht") = false,
          py::arg("rht_block_size") = 16,
          py::arg("with_random_sign_mask") = true,
          py::arg("rng_seed") = 1234ULL,
          py::arg("rng_subsequence") = 0ULL);
}
