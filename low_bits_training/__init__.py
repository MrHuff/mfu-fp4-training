#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os
import sys
import types

_LIGHT_IMPORT = os.environ.get("LBT_LIGHT_IMPORT", "0") == "1"

if not _LIGHT_IMPORT:
    from torch._inductor.kernel.flex.common import construct_strides, maybe_realize

    # Patching `flex_attention` in Torch, for TorchAO to be compatible with unstable Torch.
    flex_attention_module = types.ModuleType(
        "flex_attention", "Module created to provide a context for tests"
    )
    flex_attention_module.__dict__.update(
        {"construct_strides": construct_strides, "maybe_realize": maybe_realize}
    )
    sys.modules["torch._inductor.kernel.flex_attention"] = flex_attention_module

# Adding torchtitan submodule to `sys.path`
# TODO: check the directory exists?
sys.path.append(os.path.dirname(__file__) + "/../torchtitan_submodule")


if _LIGHT_IMPORT:
    __all__ = []
else:
    # Import device patching first.
    from . import device_patch as device_patch  # noqa: F401, E402

    if os.environ.get("PATCH_CPU"):
        device_patch.patch_cpu_device()

    # Apply logging patch early
    from . import logger as _lbt_logger  # noqa F401

    from .config import JobConfig, ConfigManager  # noqa: F401, E402

    from . import converters  # noqa: F401, E402
    from . import profiling  # noqa: F401, E402
    from . import datasets  # noqa: F401, E402
    from . import metrics  # noqa: F401, E402
    from . import models  # noqa: F401, E402
    from . import ema_checkpoint  # noqa: F401, E402
    from .checkpoint_cuda_cache import (  # noqa: E402
        install_checkpoint_cuda_cache_release,
    )

    install_checkpoint_cuda_cache_release()
    from . import compat  # noqa: F401, E402
    from . import trainer  # noqa: F401, E402

    from .quantization import (  # noqa: F401, E402
        ModelConverter,
        register_model_converter,
    )

    from . import umup as umup  # noqa: F401, E402

    from . import optimizer  # noqa: F401, E402
    from . import lr_scheduler  # noqa: F401, E402

    from .layer_stats import LayerStatsConverter as LayerStatsConverter  # noqa: F401, E402
    from .models import FusedLinearConverter as FusedLinearConverter  # noqa: F401, E402

    from . import analysis  # noqa: F401, E402

    try:
        from . import generate  # noqa: F401, E402
        from . import evaluation  # noqa: F401, E402
    except (ModuleNotFoundError, ImportError) as error:
        import warnings

        warnings.warn(
            "Unable to load evaluation tools - make sure to install "
            f"the package with pip install .[evaluate]. Error was: {error}"
        )

    from . import experiments  # noqa: F401, E402
