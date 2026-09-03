# syntax=docker/dockerfile:1
# The tag is retained for readability; the digest is the immutable contract in
# release/container_dependency_lock.json.
FROM nvcr.io/nvidia/pytorch:25.10-py3@sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee

ENV CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=2 \
    NVTE_CUDA_ARCHS=100a \
    NVTE_FRAMEWORK=pytorch \
    NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/mfu:/opt/mfu/torchtitan_submodule

WORKDIR /opt/mfu
COPY . /opt/mfu

# The NGC image supplies the recorded Python/CUDA toolchain.  Bootstrap uses
# only the vendored Transformer Engine source and disables package indexes,
# dependency resolution, and build isolation.  FP4 runtime kernels are built
# after container start because their ABI gate requires an attached SM100 GPU.
RUN scripts/release/bootstrap.sh --install-vendored

CMD ["/bin/bash"]
