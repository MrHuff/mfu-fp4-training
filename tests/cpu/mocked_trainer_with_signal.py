#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import time
import logging
from low_bits_training.signal_handler import SignalHandler

logger = logging.getLogger()


class MockedTrainerWithSignalHandler:
    """A class which implements a similar pattern as the TorchTitans Trainer
    but does no compute. It is used to test that the SignalHandler will correctly
    dump a checkpoint when SLURM sends kill and requeue signals.

    This script is used by:
    tests/cpu/test_signal_handler.py::test_trainer_integration_handles_multiple_signals
    """

    def __init__(self):
        self.max_steps = 20
        self.step_count = 0
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[titan] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.signal_handler = SignalHandler(logger=logger)
        self.signal_handler.register_handlers()
        print(
            "SUBPROCESS_LOG: Trainer initialized, SignalHandler registered.", flush=True
        )

    def train_step(self):
        print(f"SUBPROCESS_LOG: train_step, current_step={self.step_count}", flush=True)
        time.sleep(0.1)  # Tiny sleep to ensure print flushes before signal check
        if self.signal_handler.is_exit_requested():
            print(
                f"SUBPROCESS_LOG: Signal detected by trainer at step {self.step_count}.",
                flush=True,
            )
            self.max_steps = self.step_count
            time.sleep(1.0)  # Simulate cleanup
            print("SUBPROCESS_LOG: clean exit complete", flush=True)

    def train(self):
        print(
            f"SUBPROCESS_LOG: Starting training, max_steps={self.max_steps}.", flush=True
        )
        while self.step_count < self.max_steps:
            self.step_count += 1
            self.train_step()
            # Send a signal to the main process to send the kill signal once we're in the training loop
            if self.step_count == 1:
                print("SUBPROCESS_LOG: READY_TO_RECEIVE_SIGNALS", flush=True)

        print(
            f"SUBPROCESS_LOG: Training loop finished. Total steps executed: {self.step_count}.",
            flush=True,
        )


if __name__ == "__main__":
    trainer = MockedTrainerWithSignalHandler()
    trainer.train()
    print("SUBPROCESS_LOG: Subprocess finished cleanly.", flush=True)
