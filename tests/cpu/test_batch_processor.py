#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import pytest
import threading
import time


from low_bits_training.generate.batcher import BatchProcessor  # noqa


class MockGenerator:
    """Mock Generator for testing BatchProcessor."""

    def __init__(self, delay=0, fail=False, fail_on_prompt=None):
        self.generate_calls = []
        self.delay = delay  # Simulate processing time
        self.fail = fail  # Whether to simulate failure
        self.fail_on_prompt = fail_on_prompt  # Fail only on specific prompts

    def generate(
        self, prompt, batch_size, settings=None, progress_bar=False, **genkwargs
    ):
        if settings is None:
            settings = {}
        settings.update(genkwargs)
        self.generate_calls.append(
            {
                "prompt": prompt,
                "batch_size": batch_size,
                "settings": settings,
            }
        )

        # Simulate processing time
        if self.delay > 0:
            time.sleep(self.delay)

        # Check if we should fail based on specific prompts
        if self.fail_on_prompt and any(p == self.fail_on_prompt for p in prompt):
            raise ValueError(f"Mock error for prompt: {self.fail_on_prompt}")

        # Check if we should fail all calls
        if self.fail:
            raise ValueError("Mock generator failure")

        # Create mock responses
        responses = []
        for p in prompt:
            responses.append({"output_text": f"Response for: {p}"})

        return {"responses": responses}


@pytest.fixture
def mock_generator():
    """Create a mock generator for testing."""
    return MockGenerator()


@pytest.fixture
def processor(mock_generator):
    """Create a BatchProcessor instance for testing."""
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=3,
        max_queue_size=10,
        processing_interval=0.05,
    )
    yield processor
    # Cleanup after test
    processor.shutdown()
    # Allow time for thread to fully shut down
    time.sleep(0.1)


def test_basic_request_response(processor):
    """Test basic request-response flow."""
    # Set up a callback to capture the response
    result = {"response": None}
    event = threading.Event()

    def callback(response):
        result["response"] = response
        event.set()

    # Submit a request
    _ = processor.submit_request(
        prompt="Test prompt",
        temperature=0.5,
        max_new_tokens=10,
        seed=42,
        result_callback=callback,
    )

    # Wait for the response
    assert event.wait(timeout=1.0), "Callback was not called within timeout"
    assert result["response"] is not None
    assert result["response"] == "Response for: Test prompt"


def test_batching_multiple_requests():
    """Test that multiple requests are batched correctly."""
    # Use a generator with delay to ensure requests are batched
    mock_generator = MockGenerator(delay=0.2)
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=3,
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Set up callbacks for multiple requests
        results = [{"response": None} for _ in range(3)]
        events = [threading.Event() for _ in range(3)]

        def make_callback(index):
            def callback(response):
                results[index]["response"] = response
                events[index].set()

            return callback

        # Submit multiple requests rapidly
        for i in range(3):
            processor.submit_request(
                prompt=f"Test prompt {i}",
                temperature=0.5,
                max_new_tokens=10,
                seed=42,
                result_callback=make_callback(i),
            )

        # Wait for all responses
        for event in events:
            assert event.wait(
                timeout=1.0
            ), "Some callbacks were not called within timeout"

        # Verify all responses were received
        for i, result in enumerate(results):
            assert result["response"] is not None
            assert result["response"] == f"Response for: Test prompt {i}"

        # Verify the generator was called only once with a batch
        assert len(mock_generator.generate_calls) == 1
        assert len(mock_generator.generate_calls[0]["prompt"]) == 3
    finally:
        processor.shutdown()
        time.sleep(0.1)


def test_error_handling():
    """Test error handling when generator fails."""
    mock_generator = MockGenerator(fail=True)
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=3,
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Set up callback to capture the error
        result = {"response": None}
        event = threading.Event()

        def callback(response):
            result["response"] = response
            event.set()

        # Submit a request that will cause an error
        processor.submit_request(
            prompt="Test prompt",
            temperature=0.5,
            max_new_tokens=10,
            seed=42,
            result_callback=callback,
        )

        # Wait for the response
        assert event.wait(timeout=1.0), "Callback was not called within timeout"
        assert "Error during generation" in result["response"]
    finally:
        processor.shutdown()
        time.sleep(0.1)


def test_selective_error_handling():
    """Test two cases:
    1. Separate batches - each prompt is processed individually and only one error is reported
    2. Same batch - both prompts are processed together and both responses show errors
    """
    # CASE 1: Separate batches - each prompt is processed individually
    mock_generator = MockGenerator(fail_on_prompt="error_prompt")
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=1,  # Force each prompt to be in its own batch
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Set up callbacks for multiple requests
        results = [{"response": None} for _ in range(2)]
        events = [threading.Event() for _ in range(2)]

        def make_callback(index):
            def callback(response):
                results[index]["response"] = response
                events[index].set()

            return callback

        # Submit one normal request and one that will cause an error
        prompts = ["normal_prompt", "error_prompt"]
        for i, prompt in enumerate(prompts):
            processor.submit_request(
                prompt=prompt,
                temperature=0.5,
                max_new_tokens=10,
                seed=42,
                result_callback=make_callback(i),
            )

        # Wait for all responses
        for event in events:
            assert event.wait(
                timeout=1.0
            ), "Some callbacks were not called within timeout"

        # Verify both callbacks were called, with error for the second one
        assert "Response for:" in results[0]["response"]
        assert "Error during generation" in results[1]["response"]
    finally:
        processor.shutdown()
        time.sleep(0.1)

    # CASE 2: Same batch - both prompts are processed together
    mock_generator = MockGenerator(fail_on_prompt="error_prompt")
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=2,  # Allow both prompts in the same batch
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Set up callbacks for multiple requests
        results = [{"response": None} for _ in range(2)]
        events = [threading.Event() for _ in range(2)]

        def make_callback(index):
            def callback(response):
                results[index]["response"] = response
                events[index].set()

            return callback

        # Submit both prompts quickly to ensure they're in the same batch
        prompts = ["normal_prompt", "error_prompt"]
        for i, prompt in enumerate(prompts):
            processor.submit_request(
                prompt=prompt,
                temperature=0.5,
                max_new_tokens=10,
                seed=42,
                result_callback=make_callback(i),
            )

        # Wait for all responses
        for event in events:
            assert event.wait(
                timeout=1.0
            ), "Some callbacks were not called within timeout"

        # When in the same batch, both callbacks should receive error messages
        # since one bad prompt causes the entire batch to fail
        assert "Error during generation" in results[0]["response"]
        assert "Error during generation" in results[1]["response"]
    finally:
        processor.shutdown()
        time.sleep(0.1)


def test_queue_full_behavior():
    """Test behavior when queue is full."""
    # Use a generator with long delay to ensure queue fills up
    mock_generator = MockGenerator(delay=1.0)
    max_queue_size = 2
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=1,
        max_queue_size=max_queue_size,  # Small queue size for testing
        processing_interval=0.01,
    )

    try:
        # Fill the queue
        for i in range(max_queue_size + 1):  # needs to be +1 as 1 gets processed
            processor.submit_request(
                prompt=f"Test prompt {i}",
                temperature=0.5,
                max_new_tokens=10,
                seed=42,
                result_callback=lambda x: None,
            )

        # Attempt to submit another request, which should block
        blocking_thread = threading.Thread(
            target=processor.submit_request,
            kwargs={
                "prompt": "Blocking prompt",
                "temperature": 0.5,
                "max_new_tokens": 10,
                "seed": 42,
                "result_callback": lambda x: None,
            },
        )

        # Start the thread but with a timeout
        blocking_thread.daemon = True
        blocking_thread.start()

        # Check if the thread is still alive after a short delay
        # This indicates it's blocked on the queue.put() call
        time.sleep(0.2)
        assert blocking_thread.is_alive(), "Request should block when queue is full"
    finally:
        processor.shutdown(wait_for_pending=False)
        blocking_thread.join(timeout=1.0)
        time.sleep(0.1)


def test_concurrent_requests():
    """Test handling of many concurrent requests."""
    # Use a generator with short delay
    mock_generator = MockGenerator(delay=0.05)
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=5,
        max_queue_size=50,
        processing_interval=0.05,
    )

    try:
        # Set up for multiple concurrent requests
        num_requests = 20
        results = [{"response": None} for _ in range(num_requests)]
        events = [threading.Event() for _ in range(num_requests)]

        def make_callback(index):
            def callback(response):
                results[index]["response"] = response
                events[index].set()

            return callback

        # Launch many requests concurrently
        threads = []
        for i in range(num_requests):
            thread = threading.Thread(
                target=processor.submit_request,
                kwargs={
                    "prompt": f"Test prompt {i}",
                    "temperature": 0.5,
                    "max_new_tokens": 10,
                    "seed": 42,
                    "result_callback": make_callback(i),
                },
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=1.0)

        # Wait for all responses
        for event in events:
            assert event.wait(
                timeout=2.0
            ), "Some callbacks were not called within timeout"

        # Verify all responses were received
        for i, result in enumerate(results):
            assert result["response"] is not None
            assert f"Test prompt {i}" in result["response"]

        # Verify batching occurred (number of calls should be less than number of requests)
        assert len(mock_generator.generate_calls) < num_requests
    finally:
        processor.shutdown()
        time.sleep(0.1)


def test_shutdown_behavior():
    """Test that shutdown properly stops the worker thread."""
    mock_generator = MockGenerator()
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=3,
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Submit a request
        event = threading.Event()
        processor.submit_request(
            prompt="Test prompt",
            temperature=0.5,
            max_new_tokens=10,
            seed=42,
            result_callback=lambda x: event.set(),
        )

        # Wait for it to complete
        assert event.wait(timeout=1.0), "Callback was not called within timeout"

        # Get the worker thread
        worker_thread = processor.worker_thread

        # Shutdown the processor
        processor.shutdown()

        # Verify the thread has stopped
        worker_thread.join(timeout=1.0)
        assert not worker_thread.is_alive(), "Worker thread should stop after shutdown"

        # Verify that new requests are not processed after shutdown
        event = threading.Event()

        with pytest.raises(RuntimeError, match="BatchProcessor has been shut down"):
            processor.submit_request(
                prompt="Test prompt after shutdown",
                temperature=0.5,
                max_new_tokens=10,
                seed=42,
                result_callback=lambda x: event.set(),
            )

        # Wait a moment - the callback should not be triggered
        assert not event.wait(timeout=0.2), "Callback should not be called after shutdown"
    finally:
        # Already shutdown, but just to be safe
        if processor.worker_thread.is_alive():
            processor.shutdown()


def test_timeout_handling():
    """Test handling of timeouts in callbacks."""
    # Create a generator that simulates very long processing
    mock_generator = MockGenerator(delay=1.0)  # 1 second delay
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=3,
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Set up a callback with a timeout
        result = {"response": None, "timeout_occurred": False}
        event = threading.Event()

        def callback_with_timeout(response):
            result["response"] = response
            event.set()

        # Start a thread to monitor timeout
        timeout_thread = threading.Thread(
            target=lambda: result.update({"timeout_occurred": not event.wait(0.5)})
        )
        timeout_thread.daemon = True
        timeout_thread.start()

        # Submit request
        processor.submit_request(
            prompt="Long processing prompt",
            temperature=0.5,
            max_new_tokens=10,
            seed=42,
            result_callback=callback_with_timeout,
        )

        # Wait for the full processing to complete
        timeout_thread.join()
        event.wait(2.0)  # Ensure we wait long enough for the actual callback

        # Verify that timeout occurred but the response eventually arrived
        assert result["timeout_occurred"], "Timeout should have been detected"
        assert result["response"] is not None, "Response should eventually arrive"
    finally:
        processor.shutdown()
        time.sleep(0.1)


@pytest.mark.xfail(
    reason="Arguments in batcher are not being handled - need to be refactored"
)
def test_parameter_handling():
    """Test that parameters are properly handled in batched requests."""
    mock_generator = MockGenerator()
    processor = BatchProcessor(
        generator=mock_generator,
        max_batch_size=3,
        max_queue_size=10,
        processing_interval=0.05,
    )

    try:
        # Submit requests with different parameters
        events = [threading.Event() for _ in range(3)]
        callbacks = [lambda x, i=i: events[i].set() for i in range(3)]

        parameters = [
            {"temperature": 0.1, "max_new_tokens": 10, "seed": 42},
            {"temperature": 0.5, "max_new_tokens": 20, "seed": 42},
            {"temperature": 0.9, "max_new_tokens": 30, "seed": 43},
        ]

        for i, params in enumerate(parameters):
            processor.submit_request(
                prompt=f"Test prompt {i}",
                temperature=params["temperature"],
                max_new_tokens=params["max_new_tokens"],
                seed=params["seed"],
                result_callback=callbacks[i],
            )

        # Wait for all responses
        for event in events:
            assert event.wait(timeout=1.0), "Callback was not called within timeout"

        # Verify parameter handling in the batch call
        assert len(mock_generator.generate_calls) == 1
        batch_call = mock_generator.generate_calls[0]

        # Verify that temperature uses the first request's value
        assert batch_call["temperature"] == parameters[0]["temperature"]

        # Verify that max_new_tokens is set to the maximum of all requests
        assert batch_call["max_new_tokens"] == max(
            p["max_new_tokens"] for p in parameters
        )

        # Verify seed handling - should be None if they differ
        if all(p["seed"] == parameters[0]["seed"] for p in parameters):
            assert batch_call["seed"] == parameters[0]["seed"]
        else:
            assert batch_call["seed"] is None
    finally:
        processor.shutdown()
        time.sleep(0.1)
