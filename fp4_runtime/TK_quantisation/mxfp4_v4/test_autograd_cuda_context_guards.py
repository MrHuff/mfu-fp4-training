from pathlib import Path

import pytest


SOURCE = (Path(__file__).parent / "tk_quantize.cu").read_text()


def _function_body(name: str) -> str:
    start = SOURCE.index(name)
    opening = SOURCE.index("{", start)
    depth = 0
    for index in range(opening, len(SOURCE)):
        if SOURCE[index] == "{":
            depth += 1
        elif SOURCE[index] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[opening : index + 1]
    raise AssertionError(f"unterminated function body for {name}")


@pytest.mark.parametrize(
    "name",
    (
        "mxfp4_quantize_for_gemm_opt_impl",
        "mxfp4_quantize_for_gemm_opt_rht_impl",
    ),
)
def test_autograd_safe_device_context_and_stream(name: str) -> None:
    body = _function_body(name)
    assert "c10::cuda::CUDAGuard device_guard(input.device())" in body
    assert "cudaSetDevice(input.get_device())" in body
    assert "at::cuda::getCurrentCUDAStream(input.get_device())" in body
    assert "at::cuda::getCurrentCUDAStream()" not in body
