// MXFP4 CCE Backward — Fused Softmax Gradient build wrapper
#include "mxfp4_cce_backward.cuh"

#ifndef TORCH_COMPILE
int main() { return 0; }
#else
#include "pyutils/torchutils.cuh"

template <typename C>
static void launch_backward(
    const at::Tensor &A, const at::Tensor &A_sc,
    const at::Tensor &B, const at::Tensor &B_sc,
    at::Tensor &grad_out, const at::Tensor &lse, const at::Tensor &targets,
    float grad_scale, int M, int N, float filter_eps = 0.0f)
{
    using G = mxfp4_cce_backward::globals<C>;
    G g {
        .A = kittens::py::tensor_to_gl<typename G::A_fp4x2_gl>(A),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl>(A_sc),
        .B = kittens::py::tensor_to_gl<typename G::B_fp4x2_gl>(B),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl>(B_sc),
        .D_out = kittens::py::tensor_to_gl<typename G::D_gl>(grad_out),
        .lse = lse.data_ptr<float>(),
        .targets = targets.data_ptr<int64_t>(),
        .grad_scale = grad_scale,
        .filter_eps = filter_eps,
        .M = M, .N = N
    };
    kittens::py::launch_kernel<C, G, mxfp4_cce_backward::backward_kernel<C>>(g);
}

using bwd_L5_SG8 = mxfp4_cce_backward::config<5, 8, true>;
using bwd_L4_SG8 = mxfp4_cce_backward::config<4, 8, true>;
using bwd_L5_SG4 = mxfp4_cce_backward::config<5, 4, true>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_L5_SG8", &launch_backward<bwd_L5_SG8>, "MXFP4 CCE backward L5 SG8");
    m.def("backward_L4_SG8", &launch_backward<bwd_L4_SG8>, "MXFP4 CCE backward L4 SG8");
    m.def("backward_L5_SG4", &launch_backward<bwd_L5_SG4>, "MXFP4 CCE backward L5 SG4");
}
#endif
