#!/bin/bash

git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
export CUDACXX=/usr/local/cuda/bin/nvcc
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=100a

