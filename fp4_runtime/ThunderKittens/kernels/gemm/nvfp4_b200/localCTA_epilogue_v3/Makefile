GPU := B200

SRC := nvfp4_localcta_gemm.cu
OUT := _C_nv_localcta_gemm_v3$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := echo "built"
CONFIG := pytorch
NVCCFLAGS := -DTORCH_COMPILE

include ../../../common.mk

NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C_nv_localcta_gemm_v3
