#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torchtitan.tools.logging
from torchtitan.tools.logging import init_logger as _tt_init_logger


_logger_initialized = False


def init_logger_that_can_be_called_multiple_times():
    """
    Prevents adding duplicate StreamHandlers to the logger, which can cause
    errors when running tests.
    """
    # Use a global variable to track if the logger has been initialized.
    # We can't rely on checking the handlers because test suites will add
    # unrelated handlers which will cause the init_logger to never be called.
    global _logger_initialized
    if not _logger_initialized:
        _tt_init_logger()
        _logger_initialized = True


# Apply the patch when this module is imported.
torchtitan.tools.logging.init_logger = init_logger_that_can_be_called_multiple_times
