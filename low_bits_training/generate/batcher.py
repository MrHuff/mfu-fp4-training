#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import threading
import queue
import time
from dataclasses import dataclass
import multiprocessing as mp
from typing import Callable, Optional, List, Dict, Any

from low_bits_training.generate.generate import (
    Generator,
    GeneratorSettings,
    DEFAULT_GENERATOR_SETTINGS,
)


@dataclass
class GenerationRequest:
    """
    A request for text generation.
    """

    prompt: str
    settings: GeneratorSettings
    request_id: int
    result_callback: Callable[[str], None]
    created_at: float = None  # Timestamp when request was created

    def __post_init__(self):
        """Initialize the timestamp when the object is created."""
        if self.created_at is None:
            self.created_at = time.time()


class BatchProcessor:
    """
    Handles batched processing of generation requests.
    """

    def __init__(
        self,
        generator: Generator,
        max_batch_size: int = 8,
        max_queue_size: int = 100,
        processing_interval: float = 0.1,
        callback_timeout: float = 30.0,
    ):
        """
        Initialize the batch processor.

        This class is used to provide thread-safe access to the generator.
        Thread safety is needed for the integration with Simple-Evals which
        expects to ping an API with multiple requests from multiple threads.

        Args:
            generator: The generator instance to use for processing
            max_batch_size: Maximum number of requests to process in a batch
            max_queue_size: Maximum size of the request queue
            processing_interval: Time interval between processing batches in seconds
            callback_timeout: Maximum time in seconds to wait for a callback to complete
        """
        self.generator: Generator = generator
        self.max_batch_size = max_batch_size
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_interval = processing_interval
        self.callback_timeout = callback_timeout
        self.lock = threading.Lock()
        self.next_request_id = 0
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        self.shutdown_flag = False
        self.shutdown_lock = threading.Lock()

    def get_request_id(self) -> int:
        """Get a unique request ID."""
        with self.lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            return request_id

    def submit_request(
        self,
        prompt: str,
        result_callback: Callable[[str], None],
        settings: Optional[GeneratorSettings] = None,
        timeout: Optional[float] = None,
        **genkwargs,
    ) -> int:
        """
        Submit a generation request to the queue.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_new_tokens: Maximum number of tokens to generate
            seed: Random seed for reproducibility
            result_callback: Callback function to receive the result
            timeout: Optional timeout in seconds for placing the request in the queue

        Returns:
            Request ID

        Raises:
            RuntimeError: If the processor has been shut down
            queue.Full: If the queue is full and timeout is specified
        """
        if settings is None:
            settings = DEFAULT_GENERATOR_SETTINGS.copy()
        if genkwargs:
            settings.update(genkwargs)
        with self.shutdown_lock:
            if self.shutdown_flag:
                raise RuntimeError("BatchProcessor has been shut down")

        request_id = self.get_request_id()
        request = GenerationRequest(
            prompt=prompt,
            settings=settings,
            request_id=request_id,
            result_callback=result_callback,
        )
        self.request_queue.put(request, block=True, timeout=timeout)
        return request_id

    def _process_queue(self):
        """Process requests from the queue in batches."""
        while not self.stop_event.is_set():
            batch = []
            try:
                # Get at least one request (blocking)
                try:
                    request = self.request_queue.get(timeout=1.0)
                    batch.append(request)
                except queue.Empty:
                    # No requests in queue, continue the loop
                    continue

                # Try to get more requests up to max_batch_size (non-blocking)
                while len(batch) < self.max_batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        batch.append(request)
                    except queue.Empty:
                        break

                if batch:
                    self._process_batch(batch)
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # If there was an error processing the batch, notify all requesters
                for req in batch:
                    try:
                        req.result_callback(
                            f"Internal error in batch processing: {str(e)}"
                        )
                    except Exception as callback_err:
                        print(
                            f"Error calling callback for request {req.request_id}: {callback_err}"
                        )
                    finally:
                        self.request_queue.task_done()

            # Short sleep to prevent CPU spinning
            time.sleep(self.processing_interval)

    def _process_batch(self, batch: List[GenerationRequest]):
        """
        Process a batch of generation requests.

        Args:
            batch: List of generation requests to process
        """
        try:
            prompts = [req.prompt for req in batch]

            # Collect and determine parameters for the batch
            settings_list = [req.settings for req in batch]

            # Use parameters from the oldest request (first in batch after sorting)
            settings = settings_list[0]

            # Call generator with the batch
            result = self.generator.generate(
                prompt=prompts,
                batch_size=len(batch),
                settings=settings,
                progress_bar=False,
            )

            # Distribute results to callbacks
            if result and "responses" in result:
                for i, req in enumerate(batch):
                    try:
                        if i < len(result["responses"]):
                            response_text = result["responses"][i].get("output_text", "")
                            self._safe_callback(req.result_callback, response_text)
                        else:
                            self._safe_callback(
                                req.result_callback, "Failed to generate a response."
                            )
                    except Exception as e:
                        print(f"Error handling result for request {req.request_id}: {e}")
                        self._safe_callback(
                            req.result_callback, f"Error handling result: {str(e)}"
                        )
            else:
                # Something went wrong, notify all requesters
                for req in batch:
                    self._safe_callback(
                        req.result_callback,
                        "Failed to generate a response: No valid result returned",
                    )
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Notify all requesters of the failure
            for req in batch:
                self._safe_callback(
                    req.result_callback, f"Error during generation: {str(e)}"
                )
        finally:
            # Mark all requests as done in the queue
            for _ in batch:
                self.request_queue.task_done()

    def _safe_callback(self, callback: Callable[[str], None], response: str):
        """
        Safely call a callback with timeout protection.

        Args:
            callback: The callback function to call
            response: The response to pass to the callback
        """
        try:
            # Run callback in a separate thread with timeout
            result = {"completed": False}

            def run_callback():
                try:
                    callback(response)
                    result["completed"] = True
                except Exception as e:
                    print(f"Error in callback: {e}")

            callback_thread = threading.Thread(target=run_callback)
            callback_thread.daemon = True
            callback_thread.start()

            # Wait for callback to complete with timeout
            callback_thread.join(timeout=self.callback_timeout)

            if not result["completed"] and callback_thread.is_alive():
                print(
                    f"Warning: Callback timed out after {self.callback_timeout} seconds"
                )
        except Exception as e:
            print(f"Error calling callback safely: {e}")

    def shutdown(self, wait_for_pending: bool = True, timeout: Optional[float] = None):
        """
        Shutdown the batch processor.

        Args:
            wait_for_pending: Whether to wait for pending requests to complete
            timeout: Maximum time to wait for worker thread to complete
        """
        # Set shutdown flag first to prevent new submissions
        with self.shutdown_lock:
            self.shutdown_flag = True

        if wait_for_pending:
            try:
                # Wait for all queued tasks to complete
                self.request_queue.join()
            except Exception as e:
                print(f"Error waiting for pending requests: {e}")
            # Signal the worker thread to stop accepting new requests
            self.stop_event.set()
        else:
            # If not waiting for pending requests, clear the queue
            try:
                while True:
                    self.request_queue.get_nowait()
                    self.request_queue.task_done()
            except queue.Empty:
                pass
            self.stop_event.set()

        # Wait for the worker thread to exit
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout if timeout is not None else 5.0)
            if self.worker_thread.is_alive():
                print("Warning: Worker thread did not exit cleanly")

    def queue_size(self) -> int:
        """Get the current size of the request queue."""
        return self.request_queue.qsize()

    def is_alive(self) -> bool:
        """Check if the worker thread is still alive."""
        return self.worker_thread.is_alive()


def _generator_worker(
    config_path: str,
    checkpoint_path: str,
    device: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
):
    """
    Worker process to run a Generator instance on a specific device.

    Args:
        config_path: Path to the model config file.
        checkpoint_path: Path to the model checkpoint.
        device: The device to run the generator on (e.g., "cuda:0").
        request_queue: Queue for receiving generation requests.
        response_queue: Queue for sending back generation results.
    """
    try:
        generator = Generator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    except Exception as e:
        response_queue.put({"error": str(e)})
        return

    response_queue.put({"status": "ready"})

    while True:
        request = request_queue.get()
        if request is None:  # Sentinel for shutdown
            break

        try:
            result = generator.generate(**request)
            response_queue.put(result)
        except Exception as e:
            response_queue.put({"error": str(e)})


class RemoteGenerator:
    """
    A proxy for a Generator instance running in a separate process.
    This allows for running multiple generators on different GPUs.
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        """
        Initializes and starts the remote generator process.

        Args:
            config_path: Path to the model config file.
            checkpoint_path: Path to the model checkpoint.
            device: The device to run the generator on.
        """
        self.request_queue = mp.get_context("spawn").Queue()
        self.response_queue = mp.get_context("spawn").Queue()
        self.process = mp.get_context("spawn").Process(
            target=_generator_worker,
            args=(
                config_path,
                checkpoint_path,
                device,
                self.request_queue,
                self.response_queue,
            ),
        )
        self.process.start()

        # Wait for the generator to be ready
        initial_response = self.response_queue.get(
            timeout=600
        )  # 10 minutes timeout for model loading
        if "error" in initial_response:
            raise RuntimeError(
                f"Failed to initialize generator on {device}: {initial_response['error']}"
            )
        if initial_response.get("status") != "ready":
            raise RuntimeError(
                f"Unexpected initial response from generator on {device}: {initial_response}"
            )

    def generate(
        self,
        prompt: str,
        settings: GeneratorSettings,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends a generation request to the remote generator and waits for the result.

        Args:
            prompt: The input prompt(s) for generation.
            settings: The settings for generation.
            **kwargs: Additional arguments for the generator's generate method.

        Returns:
            The generation result from the remote generator.
        """
        request = {"prompt": prompt, "settings": settings, **kwargs}
        self.request_queue.put(request)
        result = self.response_queue.get()
        if "error" in result:
            raise RuntimeError(f"Error during generation: {result['error']}")
        return result

    def shutdown(self):
        """Shuts down the remote generator process."""
        self.request_queue.put(None)
        self.process.join(timeout=60)
        if self.process.is_alive():
            self.process.terminate()
