#pragma once

// The clustered kernels retire TMA stores and tensor memory before publishing
// the dependency arrival, and consumers wait before provisioning tensor
// memory. Keep that corrected overlap enabled for production; the compile-time
// override remains available as a diagnostic fallback.
#ifndef MXFP4_GEMM_DEFAULT_USE_PDL
#define MXFP4_GEMM_DEFAULT_USE_PDL 1
#endif

namespace mxfp4_launch {

inline constexpr bool default_use_pdl = MXFP4_GEMM_DEFAULT_USE_PDL != 0;

} // namespace mxfp4_launch
