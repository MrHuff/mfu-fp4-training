// ================================================================
// MXFP4 Fused Cross-Entropy — Compilation unit + PyBind11
// ================================================================
#include "mxfp4_cce.cuh"

#ifndef TORCH_COMPILE

// Standalone mode not implemented — CCE is PyTorch-only
int main() { return 0; }

#else

#include "pyutils/torchutils.cuh"

template <typename C>
static void launch_mxfp4_cce(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &lse, at::Tensor &neg_logit,
    const at::Tensor &targets, at::Tensor &D_scratch,
    int M, int N
) {
    using G = mxfp4_cce::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_scratch = kittens::py::tensor_to_gl<typename G::D_gl>(D_scratch),
        .lse = lse.data_ptr<float>(),
        .neg_correct_logit = neg_logit.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .M = M,
        .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce::kernel<C>>(g);
}

void mxfp4_cce_entrypoint(
    const at::Tensor &A,       // [M_pad, K/2] uint8
    const at::Tensor &A_sc,    // [M_pad/128, K/128, 32, 16] uint8 E8M0
    const at::Tensor &B,       // [N_pad, K/2] uint8
    const at::Tensor &B_sc,    // [N_pad/128, K/128, 32, 16] uint8 E8M0
    at::Tensor &lse,           // [M] float32 — initialized to -inf
    at::Tensor &neg_logit,     // [M] float32 — initialized to 0
    const at::Tensor &targets, // [M] int64
    at::Tensor &D_scratch,     // scratch bf16 for TMA pipeline pacing
    int M,                     // actual M (unpadded)
    int N                      // actual N (unpadded)
) {
    if (N > 64000) {
        // Large vocab (128K+): Nb=128, LOAD=4, EPI=4, SG=4 — ~8% faster
        using C = mxfp4_cce::config<128, 4, 4, 4, 2, false>;
        launch_mxfp4_cce<C>(A, A_sc, B, B_sc, lse, neg_logit, targets, D_scratch, M, N);
    } else {
        // Small vocab (≤64K): Nb=256, LOAD=5, EPI=8, SG=8
        using C = mxfp4_cce::config<256, 5, 8, 8, 2, false>;
        launch_mxfp4_cce<C>(A, A_sc, B, B_sc, lse, neg_logit, targets, D_scratch, M, N);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_cce", &mxfp4_cce_entrypoint,
          "MXFP4 Fused Cross-Entropy (no materialized logits)",
          pybind11::arg("A"), pybind11::arg("A_sc"),
          pybind11::arg("B"), pybind11::arg("B_sc"),
          pybind11::arg("lse"), pybind11::arg("neg_logit"),
          pybind11::arg("targets"), pybind11::arg("D_scratch"),
          pybind11::arg("M"), pybind11::arg("N"));
}

#endif
