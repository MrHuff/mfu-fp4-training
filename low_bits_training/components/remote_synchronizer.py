#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import subprocess
import threading
import multiprocessing as mp
from pathlib import Path

from typing import Optional, Callable, List, Tuple, Any

from datetime import datetime
from enum import IntEnum
import time

import torch.distributed

from ..config import JobConfig

from torchtitan.tools.logging import logger


class RemoteSynchronizerStatus(IntEnum):
    """Remote synchronizer process status."""

    INIT = 0  # Synchronizer initialization.
    RUNNING = 1  # Running in background process.
    EXITING = 2  # Exiting: finishing the last sync.
    EXITED = 3  # Synchronizer process exited.


def s3_sync_build_command(local_dir: str, remote_dir: str, dryrun: bool) -> list[str]:
    """Build AWS s3 sync command with all options."""
    cmd = ["aws", "s3", "sync", local_dir, remote_dir]
    cmd.append("--no-progress")
    if dryrun:
        cmd.append("--dryrun")
    return cmd


def s3_sync_worker(local_dir: str, remote_dir: str, dryrun: bool):
    """Worker function that running s3 sync in a separate process.

    Args:
        local_dir: Local directory.
        remote_dir: Remote s3 directory to sync to.
        dryrun: Only dryrun.
    Returns:
        (success, duration, raw process output).
    """
    start_time = datetime.now()
    cmd = s3_sync_build_command(local_dir, remote_dir, dryrun=dryrun)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        success = result.returncode == 0
        return success, duration, result

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Remote S3 synchronization process error: {e}")
        result = subprocess.CompletedProcess(
            cmd,
            returncode=-1,
            stdout="",
            stderr=f"{type(e).__name__}: {e}",
        )
        return False, duration, result


def s3_sync_parse_result(result) -> List[Tuple[str, str]]:
    """Same manual parsing of `aws s3 sync ...` output to extract list of files sync."""

    def _parse_line(v: str):
        v = v.replace("(dryrun)", "").strip()
        if v.startswith("upload: "):
            vals = v.split()
            return (vals[-3], vals[-1])
        else:
            return (None, None)

    lines = result.stdout.splitlines()
    lines = [v.strip() for v in lines if len(v.strip())]
    files = [_parse_line(v) for v in lines]
    return files


class RemoteSynchronizer:
    """
    Synchronizes a local directory to an S3 bucket using aws s3 sync in a separate process.

    The synchronization process is running permanently in the background, periodically calling
    `aws s3 sync` to synchronize with a remote location any new file.
    """

    def __init__(self, local_dir: str, remote_dir: str, secs_between_syncs: int = 30):
        """
        Initialize the RemoteSynchronizer.

        Args:
            local_dir: Local directory path to sync
            remote_dir: S3 bucket path (e.g., 'OBJECT_STORE_URI')
            secs_between_syncs: Number of seconds between aws s3 sync calls.
        """
        local_dir = Path(local_dir).resolve()
        if not local_dir.exists():
            raise ValueError(f"Local directory does not exist: {local_dir}")

        self.local_dir = str(local_dir)
        self.remote_dir = str(remote_dir)

        # Sleeping between aws s3 sync. call.
        # Avoid calling to often to hammer the API with dryruns.
        self._secs_between_syncs = int(secs_between_syncs)

        self._thread: Optional[threading.Thread] = None
        # Process sync. variables.
        self._status_loop = mp.Value("i", int(RemoteSynchronizerStatus.INIT))

    def __del__(self):
        self.stop_non_blocking()

    def _synchronize(self, callback: Optional[Callable] = None):
        """Synchronize using aws s3 sync in a separate process."""
        # Start with a dryrun, to get an estimate of files to sync.
        success, duration, result = s3_sync_worker(
            str(self.local_dir), str(self.remote_dir), dryrun=True
        )
        if not success:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Error, could not synchronize '{self.local_dir}' with "
                f"'{self.remote_dir}'. {details}"
            )
        files = s3_sync_parse_result(result)
        if len(files) == 0:
            return

        logger.info(
            f"Synchronising {len(files)} files to '{self.remote_dir}'. S3 sync check duration: {duration:.3f} seconds."
        )
        # Proper sync. of new files.
        success, _, result = s3_sync_worker(
            str(self.local_dir), str(self.remote_dir), dryrun=False
        )
        if callback is not None:
            callback(success, files)

    def _synchronize_loop(
        self,
        status_loop: Any,
        callback: Optional[Callable] = None,
    ):
        """Remote synchronization loop.

        Relaunching `aws s3 sync ...` periodically between local and remote dirs.

        Args:
            status_loop: mp.Value loop sync. process status.
        """

        def _is_loop_running():
            return status_loop.value == int(RemoteSynchronizerStatus.RUNNING)

        while _is_loop_running():
            try:
                self._synchronize(callback)
            except Exception:
                logger.exception(
                    "Remote synchronization attempt failed; retrying after the "
                    "configured interval."
                )
            for _ in range(self._secs_between_syncs):
                time.sleep(1.0)
                if not _is_loop_running():
                    break
        # Last sync. call to make sure no-data left behind!
        try:
            self._synchronize(callback)
        except Exception:
            logger.exception("Final remote synchronization attempt failed.")
        finally:
            status_loop.value = int(RemoteSynchronizerStatus.EXITED)

    def start(self, callback: Optional[Callable[[bool, str], None]] = None):
        """
        Start the synchronization in a background process.

        Args:
            callback: Optional callback at every synchronization.
        """

        def _callback_default(success: bool, files: Any):
            if success:
                logger.info(
                    f"✓ Sync. completed of {len(files)} files to '{self.remote_dir}'."
                )
            else:
                logger.error(
                    f"✗ Sync. failed of {len(files)} files to '{self.remote_dir}'."
                )

        if self.is_running():
            raise RuntimeError("Remote synchronization is already running.")

        callback = callback or _callback_default

        self._status_loop.value = int(RemoteSynchronizerStatus.RUNNING)
        self._thread = threading.Thread(
            target=self._synchronize_loop,
            args=(self._status_loop, callback),
        )
        self._thread.start()
        logger.info(
            f"Starting synchronization process between '{self.local_dir}' and '{self.remote_dir}'."
        )

    def status(self) -> RemoteSynchronizerStatus:
        """Get the remote synchronizer process status."""
        return RemoteSynchronizerStatus(self._status_loop.value)

    def is_running(self) -> bool:
        """Check if the sync process is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def stop_non_blocking(self) -> bool:
        """Non-blocking stop signal.

        A last synchronization is launched, and then the sync. process is exiting.
        """
        self._status_loop.value = int(RemoteSynchronizerStatus.EXITING)

    def stop(self, timeout: float = 30.0) -> bool:
        """
        Stop the sync process gracefully.

        Args:
            timeout: Time to wait before forcing termination
        """
        if not self.is_running():
            return

        self.stop_non_blocking()
        # Wait until the synchronize loop has exited.
        logger.info(
            f"Waiting for remote synchronization to '{self.remote_dir}' to finish."
        )
        # TODO: support timeout on this loop, to avoid blocking teardown for too long.
        # TODO: check what happens to partial uploads when `aws s3 sync ...` is stopped in the middle.
        while self._status_loop.value == RemoteSynchronizerStatus.EXITING:
            logger.info("Shutting down remote synchronization process...")
            time.sleep(1.0)

        logger.info(f"Stopping remote synchronization process to '{self.remote_dir}'.")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
        logger.info(
            f"Remote synchronization process to '{self.remote_dir}' closed down successfully."
        )


def is_remote_synchronizer_active(job_config: JobConfig):
    """Is the remote synchronizer active on this process?

    i.e. `remote_folder` should be set & LOCAL_RANK=0.
    """
    local_rank = torch.distributed.get_node_local_rank(0)
    return (local_rank == 0) and (job_config.job.remote_folder is not None)
