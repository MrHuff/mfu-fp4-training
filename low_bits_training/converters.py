#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
import torch.nn as nn

from typing import List, Union

from torchtitan.models.llama3 import Transformer
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from low_bits_training.config import JobConfig


class Bfloat16Converter(ModelConverter):
    """
    Make sure the model is in bfloat16
    """

    @staticmethod
    def llama3_gc(model: Transformer):
        """
        Convert Meta's and our Llama implementation. We can't use model.to(...)
        because the freq_cis tensor is of type complex64. Torch's behaviour is
        to drop the complex part during the cast.
        """
        model.layers.to(dtype=torch.bfloat16)
        model.tok_embeddings.to(dtype=torch.bfloat16)
        model.norm.to(dtype=torch.bfloat16)
        model.output.to(dtype=torch.bfloat16)

    registered_conversions = {
        "llama3": llama3_gc,
        "llama3_gc": llama3_gc,
        "nemotron_h_gc": llama3_gc,
        "nvpaper_transformer_gc": llama3_gc,
    }

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.model_flavor = job_config.model.name
        self.conversion_function = self.registered_conversions[self.model_flavor]

    def convert(self, model: nn.Module):
        self.conversion_function(model)

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(Bfloat16Converter, "bfloat16")


class Float32MasterParamsConverter(ModelConverter):
    """Keep optimizer-owned parameters in FP32 while mixed precision handles compute.

    This converter must run after structural/quantization converters so newly fused
    parameters are promoted as well. Buffers are deliberately untouched; in
    particular, Llama's complex RoPE buffer must remain complex64.
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.compute_dtype = job_config.training.mixed_precision_param

    def convert(self, model: nn.Module):
        promoted = 0
        promoted_numel = 0
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                # Conversion happens before FSDP and optimizer construction, so
                # preserving the Parameter object also preserves any weight ties.
                parameter.data = parameter.data.to(dtype=torch.float32)
                promoted += 1
                promoted_numel += parameter.numel()

        logger.info(
            "Promoted %d parameters (%d elements) to FP32 master storage; "
            "mixed-precision compute dtype=%s",
            promoted,
            promoted_numel,
            self.compute_dtype,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(Float32MasterParamsConverter, "fp32_master")


def ensure_fp32_master_params(job_config: JobConfig) -> None:
    """Make FP32 optimizer storage the default for BF16 training recipes."""

    converters = list(job_config.model.converters or [])
    existing = "fp32_master" in converters
    uses_bf16_storage = (
        job_config.training.dtype == "bfloat16" or "bfloat16" in converters
    )

    if existing:
        # Structural converters can replace parameters, so the master promotion
        # must always be the final conversion.
        job_config.model.converters = [
            converter for converter in converters if converter != "fp32_master"
        ] + ["fp32_master"]
        return

    if not uses_bf16_storage:
        return

    if not job_config.training.enable_fp32_master_params:
        logger.warning(
            "Full-BF16 optimizer storage explicitly enabled; small parameter "
            "updates may round away"
        )
        return

    job_config.model.converters = [*converters, "fp32_master"]
    logger.info(
        "Appended fp32_master to prevent full-BF16 optimizer storage; "
        "BF16 remains the mixed-precision compute dtype"
    )
