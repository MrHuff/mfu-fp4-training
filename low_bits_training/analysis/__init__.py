# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
# Set of tools to analyse checkpoints and results coming out of low-bits-training.

import os


# Standalone capture, conversion, and replay tools deliberately use the
# lightweight package mode. Do not pull the training stack (and therefore
# TorchTitan) into those tools just by importing an analysis leaf module.
if os.environ.get("LBT_LIGHT_IMPORT", "0") != "1":
    from . import stream_checkpoints as stream_checkpoints
    from . import tensor_statistics as tensor_statistics
