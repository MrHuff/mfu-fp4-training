"""Regression coverage for grouped-quant TMA descriptor staging."""

import os
from pathlib import Path
import sys

import pytest
import torch


if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import _tk_quant_v5 as tkq
except ImportError as exc:
    pytest.skip(str(exc), allow_module_level=True)


def _tensor_leaves(result):
    leaves = []
    for item in result[:7]:
        leaves.extend(item if isinstance(item, list) else [item])
    return leaves


def _raw_bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


def test_qkv_grouped_tma_staging_is_call_local():
    """Queue all Llama-8B QKV weight shapes before checking outputs."""
    torch.manual_seed(1234)
    weight = torch.randn(
        (6144, 4096), device="cuda", dtype=torch.bfloat16
    )
    splits = [4096, 1024, 1024]

    reference = tkq.tk_group_quantize_for_gemm(weight, splits)
    torch.cuda.synchronize()

    if hasattr(torch.cuda, "_sleep"):
        # Keep the copy engine behind existing stream work while Python queues
        # the descriptor copies for all transformer layers.
        torch.cuda._sleep(20_000_000)

    queued = []
    for _ in range(32):
        result = tkq.tk_group_quantize_for_gemm(weight, splits)
        assert result[7].device.type == "cpu"
        assert result[7].is_pinned()
        queued.append(result[:7])
    torch.cuda.synchronize()

    reference_leaves = _tensor_leaves(reference)
    for call_index, result in enumerate(queued):
        for tensor_index, (actual, expected) in enumerate(
            zip(_tensor_leaves(result), reference_leaves, strict=True)
        ):
            assert torch.equal(_raw_bytes(actual), _raw_bytes(expected)), (
                f"grouped quant call {call_index}, tensor {tensor_index} "
                "differs from the synchronized reference"
            )


def test_qkv_two_pass_matches_legacy_quantization():
    """The barrier-free launch must preserve every production QKV payload."""
    torch.manual_seed(2026)
    weight = torch.randn(
        (6144, 4096), device="cuda", dtype=torch.bfloat16
    )
    splits = [4096, 1024, 1024]
    old_policy = os.environ.get("USE_TK_GROUP_QUANT_TWO_PASS")
    try:
        os.environ["USE_TK_GROUP_QUANT_TWO_PASS"] = "0"
        legacy = tkq.tk_group_quantize_for_gemm(weight, splits)
        torch.cuda.synchronize()
        os.environ["USE_TK_GROUP_QUANT_TWO_PASS"] = "1"
        two_pass = tkq.tk_group_quantize_for_gemm(weight, splits)
        torch.cuda.synchronize()
    finally:
        if old_policy is None:
            os.environ.pop("USE_TK_GROUP_QUANT_TWO_PASS", None)
        else:
            os.environ["USE_TK_GROUP_QUANT_TWO_PASS"] = old_policy

    for tensor_index, (actual, expected) in enumerate(
        zip(_tensor_leaves(two_pass), _tensor_leaves(legacy), strict=True)
    ):
        assert torch.equal(_raw_bytes(actual), _raw_bytes(expected)), (
            f"two-pass grouped quant tensor {tensor_index} differs from legacy"
        )
