// ================================================================
// NVFP4 Fused Cross-Entropy — Compilation unit + PyBind11
// ================================================================
#include "nvfp4_cce.cuh"

#ifndef TORCH_COMPILE

// Standalone mode not implemented — CCE is PyTorch-only
int main() { return 0; }

#else

#include "pyutils/torchutils.cuh"

template <typename C>
static void launch_nvfp4_cce(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &lse, at::Tensor &neg_logit,
    const at::Tensor &targets, at::Tensor &D_scratch,
    int M, int N
) {
    using G = nvfp4_cce::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(A_sc, 1, A_sc.dim() == 2 ? A_sc.size(0)/128 : A_sc.size(0), A_sc.dim() == 2 ? A_sc.size(1)/4 : A_sc.size(1), 256),
        .A_sc_global = kittens::py::tensor_to_gl<typename G::A_sc_global_gl>(A_sc_global),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(B_sc, 1, B_sc.dim() == 2 ? B_sc.size(0)/128 : B_sc.size(0), B_sc.dim() == 2 ? B_sc.size(1)/4 : B_sc.size(1), 256),
        .B_sc_global = kittens::py::tensor_to_gl<typename G::B_sc_global_gl>(B_sc_global),
        .D_scratch = kittens::py::tensor_to_gl<typename G::D_gl>(D_scratch),
        .lse = lse.data_ptr<float>(),
        .neg_correct_logit = neg_logit.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .M = M,
        .N = N
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce::kernel<C>>(g);
}

void nvfp4_cce_entrypoint(
    const at::Tensor &A,           // [M_pad, K/2] uint8
    const at::Tensor &A_sc,        // [M_pad/128, K/64, 4, 256] half
    const at::Tensor &A_sc_global, // [1] float32
    const at::Tensor &B,           // [N_pad, K/2] uint8
    const at::Tensor &B_sc,        // [N_pad/128, K/64, 4, 256] half
    const at::Tensor &B_sc_global, // [1] float32
    at::Tensor &lse,               // [M] float32 — initialized to -inf
    at::Tensor &neg_logit,         // [M] float32 — initialized to 0
    const at::Tensor &targets,     // [M] int64
    at::Tensor &D_scratch,         // scratch bf16 for TMA pipeline pacing
    int M,                         // actual M (unpadded)
    int N                          // actual N (unpadded)
) {
    if (N > 64000) {
        // Large vocab (128K+): Nb=128, LOAD=4, EPI=4, SG=4
        using C = nvfp4_cce::config<128, 4, 4, 4, 2, false>;
        launch_nvfp4_cce<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, lse, neg_logit, targets, D_scratch, M, N);
    } else {
        // Small vocab (≤64K): Nb=256, LOAD=5, EPI=8, SG=8
        using C = nvfp4_cce::config<256, 5, 8, 8, 2, false>;
        launch_nvfp4_cce<C>(A, A_sc, A_sc_global, B, B_sc, B_sc_global, lse, neg_logit, targets, D_scratch, M, N);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_cce", &nvfp4_cce_entrypoint,
          "NVFP4 Fused Cross-Entropy (no materialized logits)",
          pybind11::arg("A"), pybind11::arg("A_sc"), pybind11::arg("A_sc_global"),
          pybind11::arg("B"), pybind11::arg("B_sc"), pybind11::arg("B_sc_global"),
          pybind11::arg("lse"), pybind11::arg("neg_logit"),
          pybind11::arg("targets"), pybind11::arg("D_scratch"),
          pybind11::arg("M"), pybind11::arg("N"));
}

#endif
