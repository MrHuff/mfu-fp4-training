#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import List, Optional
import signal
import logging


class SignalHandler:
    """
    Manages signal handling for graceful training shutdown.

    Sending a SIGTERM or SIGINT will cause the trainer to save a final checkpoint before exiting.

    - Use `scancel JOBID` or `kill PID` or `<C-C>`.
    - Use `scontrol requeue JOBID` to exit and requeue the job
    - Use `#SBATCH --signal=B:TERM@60` to send a SIGTERM 60 seconds before SLURM kills the job.
      this allows the job to save a checkpoint before being killed.

    NOTE: We can't use SIGUSR1 to terminate jobs as these aren't forwarded to the child
    processes by `torchrun` for some reason.

    Args:
        signals: List of signals to handle. Defaults to [signal.SIGTERM, signal.SIGINT].
          - SIGTERM (15): Termination signal, sent by `kill` or `scancel`. Exit code 0.
          - SIGINT (2): Keyboard interrupt signal. Exit code 1.
    """

    def __init__(
        self,
        signals: Optional[List[signal.Signals]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if signals is None:
            signals = [signal.SIGTERM, signal.SIGINT]
        self._signals: List[signal.Signals] = signals
        self._exit_requested: bool = False
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

    def _handle_signal(self, signum, frame):
        """Internal signal handler - sets the exit flag."""
        if not self._exit_requested:
            self._logger.warning(f"Received signal {signum}. Requesting exit.")
            self._exit_requested = True

    def register_handlers(self):
        """
        Registers signal handlers. Once registered check `is_exit_requested()`
        to see if the signal has been received.
        """
        for sig in self._signals:
            signal.signal(sig, self._handle_signal)

    def is_exit_requested(self) -> bool:
        return self._exit_requested
