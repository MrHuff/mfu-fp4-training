#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os

_LIGHT_IMPORT = os.environ.get("LBT_QUANTIZATION_LIGHT_IMPORT", "0") == "1"

if not _LIGHT_IMPORT:
    import torch
    import torchao

    from torchtitan.tools.logging import logger
    from torchtitan.protocols.model_converter import ModelConverter, register_model_converter  # noqa: F401
    from .mxfp import MXFPConverter, MXLinearGeneral  # noqa: F401

    try:
        # May fail on Aarch64 if using CPU TorchAO package.
        import torchao.prototype.mx_formats.constants

        try:
            # Keep backward compatibility on FP4 E2M1 dtype.
            from torchao.prototype.mx_formats.constants import DTYPE_FP4  # noqa: F401
        except ImportError:
            torchao.prototype.mx_formats.constants.DTYPE_FP4 = torch.float4_e2m1fn_x2
            torchao.prototype.mx_formats.mx_tensor.DTYPE_FP4 = torch.float4_e2m1fn_x2

        from .mxfp import MXFPConverter, MXLinearGeneral, mxfp_dtypes  # noqa: F401

    except ImportError as e:
        logger.warning(f"Error importing TorchAO MXFP quantization: {e}")

    # TE-native FP4 converter (register 'fp4' converter)
    try:
        from .fp4_converter import FP4Converter  # noqa: F401
    except ImportError as e:
        logger.warning(f"Error importing FP4 converter: {e}")

    try:
        from .mxfp4_tk_converter import MXFP4TKConverter  # noqa: F401
    except ImportError as e:
        logger.warning(f"Error importing MXFP4 TK converter: {e}")

    try:
        from .mixed_fp4_converter import MixedFP4LocalCTAMXFP4Converter  # noqa: F401
    except ImportError as e:
        logger.warning(f"Error importing mixed FP4 converter: {e}")
