// ================================================================
// MXFP4 Fused Cross-Entropy — Config Sweep Build
// Tests multiple GEMM configs: Nb, LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH, SUPERGROUP
// ================================================================
#include "mxfp4_cce.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else

#include "pyutils/torchutils.cuh"

// Helper: launch a specific CCE config
template <typename C>
void run_mxfp4_cce(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Config 0: Current baseline — Nb=256, EPI=8, SG=4
    m.def("mxfp4_cce_0", &run_mxfp4_cce<mxfp4_cce::config<256, 5, 8, 4, 2, false>>,
          "Nb=256 LOAD=5 EPI=8 SG=4 DT=2");

    // Config 1: Nb=128 — EPI=4, halves CCE iterations
    m.def("mxfp4_cce_1", &run_mxfp4_cce<mxfp4_cce::config<128, 5, 4, 4, 2, false>>,
          "Nb=128 LOAD=5 EPI=4 SG=4 DT=2");

    // Config 2: Nb=256, SG=8
    m.def("mxfp4_cce_2", &run_mxfp4_cce<mxfp4_cce::config<256, 5, 8, 8, 2, false>>,
          "Nb=256 LOAD=5 EPI=8 SG=8 DT=2");

    // Config 3: Nb=128, SG=8
    m.def("mxfp4_cce_3", &run_mxfp4_cce<mxfp4_cce::config<128, 5, 4, 8, 2, false>>,
          "Nb=128 LOAD=5 EPI=4 SG=8 DT=2");

    // Config 4: Nb=256, LOAD=4, EPI=8
    m.def("mxfp4_cce_4", &run_mxfp4_cce<mxfp4_cce::config<256, 4, 8, 4, 2, false>>,
          "Nb=256 LOAD=4 EPI=8 SG=4 DT=2");

    // Config 5: Nb=128, LOAD=4
    m.def("mxfp4_cce_5", &run_mxfp4_cce<mxfp4_cce::config<128, 4, 4, 4, 2, false>>,
          "Nb=128 LOAD=4 EPI=4 SG=4 DT=2");

    // Config 6: Nb=256, SG=12
    m.def("mxfp4_cce_6", &run_mxfp4_cce<mxfp4_cce::config<256, 5, 8, 12, 2, false>>,
          "Nb=256 LOAD=5 EPI=8 SG=12 DT=2");

    // Config 7: Nb=128, SG=12
    m.def("mxfp4_cce_7", &run_mxfp4_cce<mxfp4_cce::config<128, 5, 4, 12, 2, false>>,
          "Nb=128 LOAD=5 EPI=4 SG=12 DT=2");
}

#endif
