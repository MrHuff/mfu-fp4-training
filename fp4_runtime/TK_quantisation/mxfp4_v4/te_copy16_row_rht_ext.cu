#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define TE_COPY16_NO_MAIN
#include "te_style_tcgen05_hadamard_quant_bench.cu"

namespace {

static constexpr int MAX_CUDA_DEVICES_LOCAL = 16;

struct H16Cache {
    std::array<void*, MAX_CUDA_DEVICES_LOCAL> ptr{};
};

static H16Cache g_h16_cache;

struct LocalDeviceInitFlags {
    std::array<bool, MAX_CUDA_DEVICES_LOCAL> initialized{};
};

template <typename InitFn>
void ensure_local_device_init(LocalDeviceInitFlags& flags, InitFn&& init_fn) {
    int dev = 0;
    auto err = cudaGetDevice(&dev);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    TORCH_CHECK(dev >= 0 && dev < MAX_CUDA_DEVICES_LOCAL,
                "unsupported CUDA device index: ", dev);
    if (!flags.initialized[dev]) {
        init_fn();
        flags.initialized[dev] = true;
    }
}

void* get_h16_matrix() {
    int dev = 0;
    auto err = cudaGetDevice(&dev);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    TORCH_CHECK(dev >= 0 && dev < MAX_CUDA_DEVICES_LOCAL, "unsupported CUDA device index: ", dev);
    if (g_h16_cache.ptr[dev] == nullptr) {
        auto h16 = te_style_tcbench::make_hadamard_host(16, false);
        void* ptr = nullptr;
        err = cudaMalloc(&ptr, h16.size() * sizeof(__nv_bfloat16));
        TORCH_CHECK(err == cudaSuccess, "cudaMalloc h16 failed: ", cudaGetErrorString(err));
        err = cudaMemcpy(ptr, h16.data(), h16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        TORCH_CHECK(err == cudaSuccess, "cudaMemcpy h16 failed: ", cudaGetErrorString(err));
        g_h16_cache.ptr[dev] = ptr;
    }
    return g_h16_cache.ptr[dev];
}

}  // namespace

void mxfp4_row_rht_tcgen05_quantize_inplace(
    torch::Tensor input,
    torch::Tensor row_fp4,
    torch::Tensor row_sc,
    int mode
) {
    using namespace te_style_tcbench;
    TORCH_CHECK(mode == 1, "tcgen05 row-RHT path currently supports MXFP4 encode mode only");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be a 2D bf16 tensor");
    const int64_t rows64 = input.size(0);
    const int64_t cols64 = input.size(1);
    TORCH_CHECK(rows64 % CHUNK_M == 0, "rows must be divisible by 128");
    TORCH_CHECK(cols64 % CHUNK_K_WIDE == 0, "cols must be divisible by 512 for tcgen05 row-RHT");
    TORCH_CHECK(cols64 % CHUNK_DIM == 0, "cols must be divisible by 128");

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);
    TORCH_CHECK(row_fp4.is_cuda() && row_fp4.is_contiguous(), "row_fp4 must be contiguous CUDA tensor");
    TORCH_CHECK(row_fp4.scalar_type() == torch::kFloat4_e2m1fn_x2,
                "row_fp4 must have dtype float4_e2m1fn_x2");
    TORCH_CHECK(row_fp4.sizes() == torch::IntArrayRef({rows64, cols64 / 2}),
                "row_fp4 has wrong shape");
    TORCH_CHECK(row_sc.is_cuda() && row_sc.is_contiguous(), "row_sc must be contiguous CUDA tensor");
    TORCH_CHECK(row_sc.scalar_type() == torch::kUInt8, "row_sc must have dtype uint8");
    TORCH_CHECK(row_sc.sizes() == torch::IntArrayRef({rows64 / 128, cols64 / 128, 32, 16}),
                "row_sc has wrong shape");

    a128_gl A_layout{reinterpret_cast<bf16*>(input.data_ptr()), nullptr, nullptr,
                     static_cast<unsigned long>(rows), static_cast<unsigned long>(cols)};
    h16_gl H_layout{reinterpret_cast<bf16*>(get_h16_matrix()), nullptr, nullptr, nullptr, nullptr};

    int sm_count = 0;
    auto err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, input.get_device());
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
    const int total_tiles = (cols / CHUNK_K_WIDE) * (rows / CHUNK_M);
    dim3 grid(total_tiles < sm_count ? total_tiles : sm_count);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;

    static LocalDeviceInitFlags init_flags;
    ensure_local_device_init(init_flags, [&] {
        cudaFuncSetAttribute(
            te_copy16_quant_kernel_pipe64<false, false, false, true, false, true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    });

    auto stream = at::cuda::getCurrentCUDAStream();
    te_copy16_quant_kernel_pipe64<false, false, false, true, false, true, false>
        <<<grid, TE_COPY16_NUM_THREADS, smem_size, stream>>>(
            A_layout,
            H_layout,
            reinterpret_cast<uint8_t*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            nullptr,
            rows,
            cols,
            true,
            1234,
            0);
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tcgen05 row-RHT launch failed: ", cudaGetErrorString(err));
}

std::tuple<torch::Tensor, torch::Tensor>
mxfp4_row_rht_tcgen05_quantize(torch::Tensor input, int mode) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be a 2D bf16 tensor");
    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    auto device = input.device();
    auto row_fp4 = torch::empty({rows, cols / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({rows / 128, cols / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    mxfp4_row_rht_tcgen05_quantize_inplace(input, row_fp4, row_sc, mode);
    return std::make_tuple(row_fp4, row_sc);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mxfp4_row_rht_rowcol_tcgen05_quantize(torch::Tensor input, int mode) {
    using namespace te_style_tcbench;
    TORCH_CHECK(mode == 1, "tcgen05 row-RHT rowcol path currently supports MXFP4 encode mode only");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2,
                "input must be a 2D bf16 tensor");
    const int64_t rows64 = input.size(0);
    const int64_t cols64 = input.size(1);
    TORCH_CHECK(rows64 % CHUNK_M == 0, "rows must be divisible by 128");
    TORCH_CHECK(cols64 % CHUNK_K_WIDE == 0, "cols must be divisible by 512 for tcgen05 rowcol");
    TORCH_CHECK(cols64 % CHUNK_DIM == 0, "cols must be divisible by 128");

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);
    auto device = input.device();
    auto row_fp4 = torch::empty({rows64, cols64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto row_sc = torch::empty({rows64 / 128, cols64 / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));
    auto col_fp4 = torch::empty({cols64, rows64 / 2},
        torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    auto col_sc = torch::empty({cols64 / 128, rows64 / 128, 32, 16},
        torch::dtype(torch::kUInt8).device(device));

    a512_gl A_layout{reinterpret_cast<bf16*>(input.data_ptr()), nullptr, nullptr,
                     static_cast<unsigned long>(rows), static_cast<unsigned long>(cols)};
    h16_gl H_layout{reinterpret_cast<bf16*>(get_h16_matrix()), nullptr, nullptr, nullptr, nullptr};
    dim3 grid(cols / CHUNK_K_WIDE, rows / CHUNK_M);
    constexpr int smem_size = MAX_SHARED_MEMORY - 1024;

    static LocalDeviceInitFlags rowcol_init_flags;
    ensure_local_device_init(rowcol_init_flags, [&] {
        cudaFuncSetAttribute(
            te_copy16_rowcol_quant_kernel<true, false, true, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    });

    auto stream = at::cuda::getCurrentCUDAStream();
    te_copy16_rowcol_quant_kernel<true, false, true, false>
        <<<grid, TE_COPY16_NUM_THREADS, smem_size, stream>>>(
            A_layout,
            H_layout,
            reinterpret_cast<uint8_t*>(row_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(row_sc.data_ptr()),
            nullptr,
            reinterpret_cast<uint8_t*>(col_fp4.data_ptr()),
            reinterpret_cast<uint8_t*>(col_sc.data_ptr()),
            nullptr,
            rows,
            cols,
            true);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tcgen05 row-RHT rowcol launch failed: ", cudaGetErrorString(err));
    return std::make_tuple(row_fp4, row_sc, col_fp4, col_sc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_row_rht_tcgen05_quantize", &mxfp4_row_rht_tcgen05_quantize,
          "MXFP4 row-RHT tcgen05 quantize with production swizzled scales");
    m.def("mxfp4_row_rht_tcgen05_quantize_inplace", &mxfp4_row_rht_tcgen05_quantize_inplace,
          "MXFP4 row-RHT tcgen05 quantize into preallocated outputs");
    m.def("mxfp4_row_rht_rowcol_tcgen05_quantize", &mxfp4_row_rht_rowcol_tcgen05_quantize,
          "MXFP4 row-RHT tcgen05 row+col quantize with production swizzled scales");
}
