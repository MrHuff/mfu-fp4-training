// NVFP4 CCE Backward v4 — Fully Fused One-Pass (BF16 output + dE GEMM)
#include "nvfp4_cce_backward_v4.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

// Full v4: outputs BF16 grad_logits AND accumulates dE via fused Phase 3
template <typename C>
static void launch_backward_v4_bf16(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps,
    // Phase 3 inputs (pass dummies if K=0):
    const at::Tensor &C_col, const at::Tensor &C_col_sc, const at::Tensor &C_col_sc_global,
    at::Tensor &dE_out, int K)
{
    using G = nvfp4_cce_backward_v4::globals<C>;

    // Dummies for unused TMA descriptors
    auto dummy_fp4_grow = A.new_empty({C::Mb/2, C::Nb/2}, A.options().dtype(at::kFloat4_e2m1fn_x2));
    auto dummy_fp4_ccol = A.new_empty({C::Nb_out/2, C::Nb/2}, A.options().dtype(at::kFloat4_e2m1fn_x2));
    auto dummy_sc  = A.new_empty({1, 1, 256}, A.options().dtype(c10::kHalf));
    auto dummy_sg  = A.new_empty({1}, A.options().dtype(c10::kFloat));
    auto dummy_dE  = A.new_empty({C::Mb/2, C::Nb_out}, A.options().dtype(c10::kBFloat16));

    bool use_p3 = (K > 0);
    at::Tensor c_col_sc_storage = dummy_sc;
    int c_col_sc_depth = 1;
    int c_col_sc_rows = 1;

    if (use_p3) {
        TORCH_CHECK(C_col.is_cuda() && C_col.is_contiguous(),
                    "Phase 3 C_col must be a contiguous CUDA tensor from tk_quantize_for_gemm(...)[2].");
        TORCH_CHECK(C_col_sc.is_cuda() && C_col_sc.is_contiguous(),
                    "Phase 3 C_col_sc must be a contiguous CUDA tensor from tk_quantize_for_gemm(...)[3].");
        TORCH_CHECK(C_col.dim() == 2,
                    "Phase 3 C_col must have shape [K, V/2] from the column-quantized output.");
        TORCH_CHECK(C_col.size(0) == K,
                    "Phase 3 C_col first dimension must equal K. Got ", C_col.size(0), " vs ", K, ".");
        TORCH_CHECK(C_col.size(1) == N / 2,
                    "Phase 3 C_col second dimension must equal V/2. Got ", C_col.size(1), " vs ", N / 2, ".");
        TORCH_CHECK(C_col_sc.dim() == 3 && C_col_sc.size(0) == K / 128 && C_col_sc.size(2) == 512,
                    "Phase 3 C_col_sc must have shape [K/128, V/64, 512] from the column-quantized output.");

        c_col_sc_depth = C_col_sc.size(0);
        c_col_sc_rows = C_col_sc.size(1);
        if (c_col_sc_rows < 2) {
            c_col_sc_storage = C_col_sc.new_zeros({c_col_sc_depth, 2, 512}, C_col_sc.options());
            c_col_sc_storage.narrow(1, 0, c_col_sc_rows).copy_(C_col_sc);
            c_col_sc_rows = 2;
        } else {
            c_col_sc_storage = C_col_sc;
        }
    }

    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            A_sc, 1,
            A_sc.dim()==2 ? A_sc.size(0)/128 : A_sc.size(0),
            A_sc.dim()==2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            B_sc, 1,
            B_sc.dim()==2 ? B_sc.size(0)/128 : B_sc.size(0),
            B_sc.dim()==2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        // Phase 3 C_col: real tensors when K>0, dummies when K=0
        .C_col = use_p3
            ? kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(C_col)
            : kittens::py::tensor_to_gl<typename G::P3_B_fp4x2_gl>(dummy_fp4_ccol),
        .C_col_sc = kittens::py::tensor_to_gl<typename G::P3_B_sc_gl, false>(
            c_col_sc_storage, 1, c_col_sc_depth, c_col_sc_rows, 256),
        .C_col_sc_global = kittens::py::tensor_to_gl<typename G::P3_B_sc_global_gl>(
            use_p3 ? C_col_sc_global : dummy_sg),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .dE_out = use_p3
            ? kittens::py::tensor_to_gl<typename G::dE_gl>(dE_out)
            : kittens::py::tensor_to_gl<typename G::dE_gl>(dummy_dE),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4_row_gl>(dummy_fp4_grow),
        .G_sc_row_ptr = nullptr,
        .G_sc_row_kgroups = 1,
        .G_sg_row = kittens::py::tensor_to_gl<typename G::G_sg_row_gl>(dummy_sg),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 1,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N, .K = K,
        .encode_centric = false
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_backward_v4::backward_kernel_v4<C>>(g);
}

// Config instantiations
using bwd_v4_bf16_L4_SG8 = nvfp4_cce_backward_v4::config<4, 8, true, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v4_bf16_L4_SG8", &launch_backward_v4_bf16<bwd_v4_bf16_L4_SG8>,
          "NVFP4 CCE backward v4 (BF16+dE fused) L4 SG8");
}
#endif
