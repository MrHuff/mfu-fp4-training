cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
project(cutlass_gemm_evaluator CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

# 1. Path to your CORE CUTLASS library headers
set(CUTLASS_INCLUDE_DIR ./cutlass/include)

# 2. Path to your CUTLASS UTILITIES headers (for HostTensor)
set(CUTLASS_UTIL_INCLUDE_DIR ./cutlass/tools/util/include)

# 3. Path to your CUTLASS REFERENCE headers (for verification)
set(CUTLASS_REF_INCLUDE_DIR ./cutlass/tools/util/reference/include)

# 4. Path to the PROFILER headers where device_allocation.h was found <--- NEW
set(CUTLASS_PROFILER_INCLUDE_DIR ./cutlass/tools/profiler/include)
set(CUTLASS_HELPER_INCLUDE_DIR ./cutlass/examples/common)


add_executable(gemm_evaluator gemm_evaluator.cu)
target_compile_options(gemm_evaluator PRIVATE "--expt-relaxed-constexpr")

# Add ALL necessary include paths to the target
target_include_directories(gemm_evaluator PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_UTIL_INCLUDE_DIR}
    ${CUTLASS_REF_INCLUDE_DIR}
    ${CUTLASS_PROFILER_INCLUDE_DIR} 
    ${CUTLASS_HELPER_INCLUDE_DIR} # <--- NEW
)
# --- FIX 2: Link against the CUDA libraries variable ---
# Use the ${CUDA_LIBRARIES} variable, which contains the path to cudart and other libs.
target_link_libraries(gemm_evaluator PRIVATE ${CUDA_LIBRARIES})
# --- IMPORTANT ---
# Set GPU architecture flags.
set(CMAKE_CUDA_ARCHITECTURES 100)