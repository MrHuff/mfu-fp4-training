#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import logging

from low_bits_training.utils import NanLossDetectingHandler


class TestStopOnNanLoss:
    def test_stop_on_nan_loss(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        nan_loss_detector = NanLossDetectingHandler()
        nan_loss_detector.setLevel(logging.INFO)
        logger.addHandler(nan_loss_detector)

        logger.info("step:  1, loss:    3.1415")
        assert not nan_loss_detector.nan_seen()

        logger.info("step:  2, loss:    nan")
        assert nan_loss_detector.nan_seen()
