// NVFP4 CCE v2 — Ping-Pong build wrapper
#include "nvfp4_cce_v2.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

template <typename C>
static void launch_pp(
    const at::Tensor &A, const at::Tensor &A_sc, const at::Tensor &A_sc_global,
    const at::Tensor &B, const at::Tensor &B_sc, const at::Tensor &B_sc_global,
    at::Tensor &lse, at::Tensor &neg_logit,
    const at::Tensor &targets, at::Tensor &D_scratch, int M, int N)
{
    using G = nvfp4_cce_v2::pp_globals<C>;
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
        .D_scratch = kittens::py::tensor_to_gl<typename G::D_gl>(D_scratch),
        .lse = lse.data_ptr<float>(),
        .neg_correct_logit = neg_logit.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, nvfp4_cce_v2::pp_kernel<C>>(g);
}

// Ping-pong configs
using pp_L4_SG8 = nvfp4_cce_v2::pp_config<4, 8, true>;
using pp_L3_SG8 = nvfp4_cce_v2::pp_config<3, 8, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pp_L4_SG8", &launch_pp<pp_L4_SG8>, "NVFP4 CCE v2 ping-pong L4 SG8");
    m.def("pp_L3_SG8", &launch_pp<pp_L3_SG8>, "NVFP4 CCE v2 ping-pong L3 SG8");
}
#endif
