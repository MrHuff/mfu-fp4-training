GPU := B200

SRC := nvfp4_b200_gemm.cu
ifeq ($(SHARED_B_DEBUG_STANDALONE),1)
OUT := nvfp4_b200_shared_b_debug.out
CMD := ./$(OUT)
NVCCFLAGS += -DSHARED_B_DEBUG_STANDALONE
CONFIG := standalone
else
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test_gemm.py # or test_quantize.py
NVCCFLAGS += -DTORCH_COMPILE
CONFIG := pytorch
endif

include ../../common.mk

SHARED_B_DEBUG_OUT := nvfp4_b200_shared_b_debug.out

shared_b_debug_runner:
	$(MAKE) SHARED_B_DEBUG_STANDALONE=1 all

shared_b_debug_run: shared_b_debug_runner
	./$(SHARED_B_DEBUG_OUT)

.PHONY: shared_b_debug_runner shared_b_debug_run
