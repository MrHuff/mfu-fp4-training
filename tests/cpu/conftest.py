#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import shared_fixtures


unit_test_checkpoint = shared_fixtures.unit_test_checkpoint
config_and_checkpoint = shared_fixtures.config_and_checkpoint
no_distribution = shared_fixtures.no_distribution
