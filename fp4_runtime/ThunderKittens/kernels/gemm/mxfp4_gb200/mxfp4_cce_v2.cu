// ================================================================
// MXFP4 Fused Cross-Entropy v2 — Ping-Pong MMA
// Pybind11 bindings + kernel dispatch
// ================================================================
#include "mxfp4_cce_v2.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else

#include "pyutils/torchutils.cuh"

template <typename C>
void run_mxfp4_cce_v2(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &lse, at::Tensor &neg_logit,
    const at::Tensor &targets, at::Tensor &D_scratch,
    int M, int N
) {
    using G = mxfp4_cce_v2::globals<C>;
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
    kittens::py::launch_kernel<C, G, mxfp4_cce_v2::kernel<C>>(g);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Non-pingpong baseline (identical to v1 Nb=128)
    m.def("no_pp_L5_SG4", &run_mxfp4_cce_v2<mxfp4_cce_v2::config<5, 4, false>>,
          "No pingpong L5 SG4");
    m.def("no_pp_L5_SG8", &run_mxfp4_cce_v2<mxfp4_cce_v2::config<5, 8, false>>,
          "No pingpong L5 SG8");

    // Pingpong configs
    m.def("pp_L5_SG4", &run_mxfp4_cce_v2<mxfp4_cce_v2::config<5, 4, true>>,
          "Pingpong L5 SG4");
    m.def("pp_L5_SG8", &run_mxfp4_cce_v2<mxfp4_cce_v2::config<5, 8, true>>,
          "Pingpong L5 SG8");
    m.def("pp_L4_SG4", &run_mxfp4_cce_v2<mxfp4_cce_v2::config<4, 4, true>>,
          "Pingpong L4 SG4");
    m.def("pp_L4_SG8", &run_mxfp4_cce_v2<mxfp4_cce_v2::config<4, 8, true>>,
          "Pingpong L4 SG8");
}

#endif
