from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.distributed.checkpoint as dcp


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvl72"))

import compare_dcp_tensors as comparator  # noqa: E402


def test_streams_matching_dcp_chunks_and_reports_row_outliers(tmp_path):
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    left = {
        "output.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "norm.weight": torch.ones(4),
    }
    right = {
        "output.weight": left["output.weight"] + 1,
        "norm.weight": torch.full((4,), 2.0),
    }
    with pytest.warns(UserWarning, match="single process"):
        dcp.save(left, checkpoint_id=left_path)
    with pytest.warns(UserWarning, match="single process"):
        dcp.save(right, checkpoint_id=right_path)

    result = comparator.run(
        SimpleNamespace(
            left=str(left_path),
            right=str(right_path),
            key_regex=[r"^(output|norm)\.weight$"],
            top_rows=2,
            output=None,
        )
    )
    by_key = {item["key"]: item for item in result["keys"]}

    assert by_key["norm.weight"]["delta"]["mean"] == -1.0
    assert by_key["output.weight"]["delta"]["mean"] == -1.0
    assert by_key["output.weight"]["delta"]["cosine"] == pytest.approx(0.9993484634)
    outliers = by_key["output.weight"]["rows"]["outliers"]
    assert [item["row"] for item in outliers["largest_left_l2"]] == [5, 4]
    assert len(outliers["largest_delta_l2"]) == 2


def test_compares_explicitly_mapped_tensor_keys(tmp_path):
    left_path = tmp_path / "left"
    right_path = tmp_path / "right"
    with pytest.warns(UserWarning, match="single process"):
        dcp.save(
            {"fused.norm_weight": torch.ones(4)},
            checkpoint_id=left_path,
        )
    with pytest.warns(UserWarning, match="single process"):
        dcp.save(
            {"norm.weight": torch.full((4,), 2.0)},
            checkpoint_id=right_path,
        )

    result = comparator.run(
        SimpleNamespace(
            left=str(left_path),
            right=str(right_path),
            key_regex=[],
            key_map=["fused.norm_weight=norm.weight"],
            top_rows=0,
            workers=2,
            output=None,
        )
    )

    item = result["keys"][0]
    assert item["key"] == "fused.norm_weight"
    assert item["right_key"] == "norm.weight"
    assert item["delta"]["mean"] == -1.0
