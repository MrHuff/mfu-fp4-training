// MXFP4 CCE Backward v2 — Fused softmax gradient + optional FP4 quantization
#include "mxfp4_cce_backward_v2.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

static at::Tensor make_unused_fp4_row_placeholder(const at::Tensor &ref) {
    return ref.new_empty({128, 64}, ref.options().dtype(c10::kFloat4_e2m1fn_x2));
}

static at::Tensor make_unused_bf16_placeholder(const at::Tensor &ref) {
    return ref.new_empty({128, 32}, ref.options().dtype(c10::kBFloat16));
}

// BF16 mode: outputs BF16 grad_logits
template <typename C>
static void launch_backward_v2_bf16(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(
            make_unused_fp4_row_placeholder(A)),
        .G_sc_row = nullptr,
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 1,
        .G_tilemask = nullptr,
        .G_tilemask_cols = 0,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

// FP4 mode: outputs quantized G in MXFP4 format
template <typename C>
static void launch_backward_v2_fp4(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            make_unused_bf16_placeholder(A)),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(G_fp4_row),
        .G_sc_row = G_sc_row.data_ptr<uint8_t>(),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 1,
        .G_tilemask = nullptr,
        .G_tilemask_cols = 0,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

template <typename C>
static void launch_backward_v2_fp4_rowonly(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            make_unused_bf16_placeholder(A)),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(G_fp4_row),
        .G_sc_row = G_sc_row.data_ptr<uint8_t>(),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 0,
        .G_tilemask = nullptr,
        .G_tilemask_cols = 0,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

template <typename C>
static void launch_backward_v2_fp4_rowonly_masked(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row,
    at::Tensor &G_tilemask,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            make_unused_bf16_placeholder(A)),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(G_fp4_row),
        .G_sc_row = G_sc_row.data_ptr<uint8_t>(),
        .G_fp4_col_ptr = nullptr,
        .G_sc_col_ptr = nullptr,
        .G_sc_col_kgroups = 0,
        .G_tilemask = G_tilemask.data_ptr<uint8_t>(),
        .G_tilemask_cols = static_cast<int>(G_tilemask.size(1)),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

template <typename C>
static void launch_backward_v2_fp4_colonly_masked(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    at::Tensor &G_tilemask,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            make_unused_bf16_placeholder(A)),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(
            make_unused_fp4_row_placeholder(A)),
        .G_sc_row = nullptr,
        .G_fp4_col_ptr = reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        .G_sc_col_ptr = reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        .G_sc_col_kgroups = static_cast<int>(A.size(0) / 128),
        .G_tilemask = G_tilemask.data_ptr<uint8_t>(),
        .G_tilemask_cols = static_cast<int>(G_tilemask.size(1)),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

template <typename C>
static void launch_backward_v2_fp4_rowcol(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            make_unused_bf16_placeholder(A)),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(G_fp4_row),
        .G_sc_row = G_sc_row.data_ptr<uint8_t>(),
        .G_fp4_col_ptr = G_fp4_col.numel() == 0 ? nullptr : reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        .G_sc_col_ptr = G_sc_col.numel() == 0 ? nullptr : reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        .G_sc_col_kgroups = G_sc_col.numel() == 0 ? 0 : static_cast<int>(A.size(0) / 128),
        .G_tilemask = nullptr,
        .G_tilemask_cols = 0,
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

template <typename C>
static void launch_backward_v2_fp4_rowcol_masked(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &G_fp4_row, at::Tensor &G_sc_row,
    at::Tensor &G_fp4_col, at::Tensor &G_sc_col,
    at::Tensor &G_tilemask,
    const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward_v2::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(
            make_unused_bf16_placeholder(A)),
        .G_fp4_row = kittens::py::tensor_to_gl<typename G::G_fp4x2_gl>(G_fp4_row),
        .G_sc_row = G_sc_row.data_ptr<uint8_t>(),
        .G_fp4_col_ptr = G_fp4_col.numel() == 0 ? nullptr : reinterpret_cast<uint8_t*>(G_fp4_col.data_ptr()),
        .G_sc_col_ptr = G_sc_col.numel() == 0 ? nullptr : reinterpret_cast<uint8_t*>(G_sc_col.data_ptr()),
        .G_sc_col_kgroups = G_sc_col.numel() == 0 ? 0 : static_cast<int>(A.size(0) / 128),
        .G_tilemask = G_tilemask.data_ptr<uint8_t>(),
        .G_tilemask_cols = static_cast<int>(G_tilemask.size(1)),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward_v2::backward_kernel_v2<C>>(g);
}

// Config instantiations
using bwd_v2_bf16_L5_SG8 = mxfp4_cce_backward_v2::config<5, 8, true, true>;
using bwd_v2_bf16_L4_SG8 = mxfp4_cce_backward_v2::config<4, 8, true, true>;
using bwd_v2_fp4_L5_SG8  = mxfp4_cce_backward_v2::config<5, 8, false, true>;
using bwd_v2_fp4_L4_SG8  = mxfp4_cce_backward_v2::config<4, 8, false, true>;
using bwd_v2_fp4_rowcol_L4_SG8 = bwd_v2_fp4_L4_SG8;
using bwd_v2_fp4_rowonly_L4_SG8 = bwd_v2_fp4_L4_SG8;
using bwd_v2_fp4_colonly_L4_SG8 = bwd_v2_fp4_L4_SG8;
// Encode-centric: ceil E8M0 exponent (QUANT_MODE=1)
using bwd_v2_fp4_enc_L4_SG8 = mxfp4_cce_backward_v2::config<4, 8, false, true, 1>;
using bwd_v2_fp4_enc_rowcol_L4_SG8 = bwd_v2_fp4_enc_L4_SG8;
using bwd_v2_fp4_enc_rowonly_L4_SG8 = bwd_v2_fp4_enc_L4_SG8;
using bwd_v2_fp4_enc_colonly_L4_SG8 = bwd_v2_fp4_enc_L4_SG8;
// Decode-centric: floor E8M0 exponent (QUANT_MODE=2)
using bwd_v2_fp4_dec_L4_SG8 = mxfp4_cce_backward_v2::config<4, 8, false, true, 2>;
using bwd_v2_fp4_dec_rowcol_L4_SG8 = bwd_v2_fp4_dec_L4_SG8;
using bwd_v2_fp4_dec_rowonly_L4_SG8 = bwd_v2_fp4_dec_L4_SG8;
using bwd_v2_fp4_dec_colonly_L4_SG8 = bwd_v2_fp4_dec_L4_SG8;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_v2_bf16_L5_SG8", &launch_backward_v2_bf16<bwd_v2_bf16_L5_SG8>,
          "MXFP4 CCE backward v2 (BF16 output) L5 SG8");
    m.def("backward_v2_bf16_L4_SG8", &launch_backward_v2_bf16<bwd_v2_bf16_L4_SG8>,
          "MXFP4 CCE backward v2 (BF16 output) L4 SG8");
    m.def("backward_v2_fp4_L5_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_L5_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE output) L5 SG8");
    m.def("backward_v2_fp4_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE output) L4 SG8");
    m.def("backward_v2_fp4_rowonly_L4_SG8", &launch_backward_v2_fp4_rowonly<bwd_v2_fp4_rowonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE row-only output) L4 SG8");
    m.def("backward_v2_fp4_colonly_masked_L4_SG8", &launch_backward_v2_fp4_colonly_masked<bwd_v2_fp4_colonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE col-only output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_rowcol_L4_SG8", &launch_backward_v2_fp4_rowcol<bwd_v2_fp4_rowcol_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE row+col output) L4 SG8");
    m.def("backward_v2_fp4_rowonly_masked_L4_SG8", &launch_backward_v2_fp4_rowonly_masked<bwd_v2_fp4_rowonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE row-only output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_rowcol_masked_L4_SG8", &launch_backward_v2_fp4_rowcol_masked<bwd_v2_fp4_rowcol_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 RTE row+col output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_enc_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_enc_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric output) L4 SG8");
    m.def("backward_v2_fp4_enc_rowonly_L4_SG8", &launch_backward_v2_fp4_rowonly<bwd_v2_fp4_enc_rowonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric row-only output) L4 SG8");
    m.def("backward_v2_fp4_enc_colonly_masked_L4_SG8", &launch_backward_v2_fp4_colonly_masked<bwd_v2_fp4_enc_colonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric col-only output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_enc_rowcol_L4_SG8", &launch_backward_v2_fp4_rowcol<bwd_v2_fp4_enc_rowcol_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric row+col output) L4 SG8");
    m.def("backward_v2_fp4_enc_rowonly_masked_L4_SG8", &launch_backward_v2_fp4_rowonly_masked<bwd_v2_fp4_enc_rowonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric row-only output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_enc_rowcol_masked_L4_SG8", &launch_backward_v2_fp4_rowcol_masked<bwd_v2_fp4_enc_rowcol_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 encode-centric row+col output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_dec_L4_SG8", &launch_backward_v2_fp4<bwd_v2_fp4_dec_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric output) L4 SG8");
    m.def("backward_v2_fp4_dec_rowonly_L4_SG8", &launch_backward_v2_fp4_rowonly<bwd_v2_fp4_dec_rowonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric row-only output) L4 SG8");
    m.def("backward_v2_fp4_dec_colonly_masked_L4_SG8", &launch_backward_v2_fp4_colonly_masked<bwd_v2_fp4_dec_colonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric col-only output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_dec_rowcol_L4_SG8", &launch_backward_v2_fp4_rowcol<bwd_v2_fp4_dec_rowcol_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric row+col output) L4 SG8");
    m.def("backward_v2_fp4_dec_rowonly_masked_L4_SG8", &launch_backward_v2_fp4_rowonly_masked<bwd_v2_fp4_dec_rowonly_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric row-only output + tile mask) L4 SG8");
    m.def("backward_v2_fp4_dec_rowcol_masked_L4_SG8", &launch_backward_v2_fp4_rowcol_masked<bwd_v2_fp4_dec_rowcol_L4_SG8>,
          "MXFP4 CCE backward v2 (FP4 decode-centric row+col output + tile mask) L4 SG8");
}
#endif
