# Makefile for mxfp4_v4 pipelined quantize kernel
NVCC := /usr/local/cuda/bin/nvcc

PYTHON := python3
PY_INC := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIB := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_VER := $(shell $(PYTHON) -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

PYBIND_INC := $(shell $(PYTHON) -m pybind11 --includes)
TORCH_INC := $(shell $(PYTHON) -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0])")
TORCH_INC2 := $(shell $(PYTHON) -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[1])")
TORCH_LIB := $(shell $(PYTHON) -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.library_paths()[0])")

SRC := tk_quantize.cu
HEADERS := mxfp4_v3_quantize.cuh ../../ThunderKittens/kernels/gemm/common/c1_rms_reduce.cuh
OUT := mxfp4_quant_v4$(EXT_SUFFIX)
BENCH_SRC := tcgen05_hadamard_quant_bench.cu
BENCH_OUT := tcgen05_hadamard_quant_bench
BENCH_TE_SRC := te_style_tcgen05_hadamard_quant_bench.cu
BENCH_TE_OUT := te_style_tcgen05_hadamard_quant_bench
TECOPY_EXT_SRC := te_copy16_row_rht_ext.cu
TECOPY_EXT_OUT := mxfp4_tecopy_rht$(EXT_SUFFIX)
MOE_COMBINE_EXT_SRC := mxfp4_moe_combine.cu
MOE_COMBINE_EXT_OUT := mxfp4_moe_combine$(EXT_SUFFIX)
MOE_PRODUCER_EXT_SRC := mxfp4_moe_producer.cu
MOE_PRODUCER_EXT_OUT := mxfp4_moe_producer$(EXT_SUFFIX)

NVCCFLAGS := -std=c++20 -O3 \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -forward-unknown-to-host-compiler \
    -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
    -DNDEBUG -lineinfo \
    -shared -fPIC \
    -diag-suppress 3189 -diag-suppress 177 \
    -D__CUDA_NO_HALF_OPERATORS__ \
    -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -DTORCH_EXTENSION_NAME=mxfp4_quant_v4 \
    -I$(PY_INC) \
    -I../nvfp4_v5 \
    $(PYBIND_INC) \
    -I$(TORCH_INC) -I$(TORCH_INC2) \
    -L$(TORCH_LIB) -L$(PY_LIB) \
    -l$(PY_VER) -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 \
    -lcuda -lcudart_static -lcudadevrt -ldl \
    -gencode arch=compute_100a,code=sm_100a

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

bench: $(BENCH_OUT)

$(BENCH_OUT): $(BENCH_SRC) $(HEADERS)
	$(NVCC) $(BENCH_SRC) -std=c++20 -O3 \
		--expt-extended-lambda --expt-relaxed-constexpr \
		-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
		-lineinfo -DKITTENS_BLACKWELL -I../../ThunderKittens/include -I../nvfp4_v5 -lcuda -lcudart_static -lcudadevrt -ldl \
		-gencode arch=compute_100a,code=sm_100a -o $(BENCH_OUT)

bench_te: $(BENCH_TE_OUT)

$(BENCH_TE_OUT): $(BENCH_TE_SRC) $(HEADERS)
	$(NVCC) $(BENCH_TE_SRC) -std=c++20 -O3 \
		--expt-extended-lambda --expt-relaxed-constexpr \
		-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
		-lineinfo -DKITTENS_BLACKWELL -I../../ThunderKittens/include -I../nvfp4_v5 -lcuda -lcudart_static -lcudadevrt -ldl \
		-gencode arch=compute_100a,code=sm_100a -o $(BENCH_TE_OUT)

tecopy_ext: $(TECOPY_EXT_OUT)

$(TECOPY_EXT_OUT): $(TECOPY_EXT_SRC) $(BENCH_TE_SRC) $(HEADERS)
	$(NVCC) $(TECOPY_EXT_SRC) -std=c++20 -O3 \
		--expt-extended-lambda --expt-relaxed-constexpr \
		-forward-unknown-to-host-compiler \
		-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
		-DNDEBUG -lineinfo -shared -fPIC \
		-diag-suppress 3189 -diag-suppress 177 \
		-DKITTENS_BLACKWELL \
		-D__CUDA_NO_HALF_OPERATORS__ \
		-D__CUDA_NO_HALF_CONVERSIONS__ \
		-D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
		-D__CUDA_NO_HALF2_OPERATORS__ \
		-DTORCH_API_INCLUDE_EXTENSION_H \
		-DTORCH_EXTENSION_NAME=mxfp4_tecopy_rht \
		-I$(PY_INC) \
		-I../../ThunderKittens/include \
		-I../nvfp4_v5 \
		$(PYBIND_INC) \
		-I$(TORCH_INC) -I$(TORCH_INC2) \
		-L$(TORCH_LIB) -L$(PY_LIB) \
		-l$(PY_VER) -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 \
		-lcuda -lcudart_static -lcudadevrt -ldl \
		-gencode arch=compute_100a,code=sm_100a -o $(TECOPY_EXT_OUT)

moe_combine_ext: $(MOE_COMBINE_EXT_OUT)

$(MOE_COMBINE_EXT_OUT): $(MOE_COMBINE_EXT_SRC)
	$(NVCC) $(MOE_COMBINE_EXT_SRC) -std=c++20 -O3 \
		--expt-extended-lambda --expt-relaxed-constexpr \
		-forward-unknown-to-host-compiler \
		-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
		-DNDEBUG -lineinfo -shared -fPIC \
		-DTORCH_API_INCLUDE_EXTENSION_H \
		-DTORCH_EXTENSION_NAME=mxfp4_moe_combine \
		-I$(PY_INC) \
		$(PYBIND_INC) \
		-I$(TORCH_INC) -I$(TORCH_INC2) \
		-L$(TORCH_LIB) -L$(PY_LIB) \
		-l$(PY_VER) -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 \
		-lcuda -lcudart_static -lcudadevrt -ldl \
		-gencode arch=compute_100a,code=sm_100a -o $(MOE_COMBINE_EXT_OUT)

moe_producer_ext: $(MOE_PRODUCER_EXT_OUT)

$(MOE_PRODUCER_EXT_OUT): $(MOE_PRODUCER_EXT_SRC) $(HEADERS)
	$(NVCC) $(MOE_PRODUCER_EXT_SRC) -std=c++20 -O3 \
		--expt-extended-lambda --expt-relaxed-constexpr \
		-forward-unknown-to-host-compiler \
		-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
		-DNDEBUG -lineinfo -shared -fPIC \
		-diag-suppress 3189 -diag-suppress 177 \
		-D__CUDA_NO_HALF_OPERATORS__ \
		-D__CUDA_NO_HALF_CONVERSIONS__ \
		-D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
		-D__CUDA_NO_HALF2_OPERATORS__ \
		-DTORCH_API_INCLUDE_EXTENSION_H \
		-DTORCH_EXTENSION_NAME=mxfp4_moe_producer \
		-I$(PY_INC) \
		-I../nvfp4_v5 \
		$(PYBIND_INC) \
		-I$(TORCH_INC) -I$(TORCH_INC2) \
		-L$(TORCH_LIB) -L$(PY_LIB) \
		-l$(PY_VER) -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 \
		-lcuda -lcudart_static -lcudadevrt -ldl \
		-gencode arch=compute_100a,code=sm_100a -o $(MOE_PRODUCER_EXT_OUT)

clean:
	rm -f $(OUT) $(BENCH_OUT) $(BENCH_TE_OUT) $(TECOPY_EXT_OUT) $(MOE_COMBINE_EXT_OUT) $(MOE_PRODUCER_EXT_OUT)

.PHONY: all clean bench bench_te tecopy_ext moe_combine_ext moe_producer_ext
