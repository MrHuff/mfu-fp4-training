from pathlib import Path
import sys

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "torchtitan_submodule"))

from torchtitan.models.llama3.infra import parallelize  # noqa: E402
from low_bits_training.models import nemotron_h  # noqa: E402


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Linear(2, 2)
        self.layers = nn.ModuleDict(
            {
                "0": nn.Linear(2, 2),
                "1": nn.Linear(2, 2),
            }
        )
        self.norm = nn.LayerNorm(2)
        self.output = nn.Linear(2, 2)


class _NemotronModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Linear(2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.norm = nn.LayerNorm(2)
        self.output = nn.Linear(2, 2)


def test_fsdp_preserves_carrier_layer_input_dtypes(monkeypatch):
    model = _Model()
    model.layers["1"]._fsdp_preserve_forward_input_dtypes = True
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(parallelize, "fully_shard", fake_fully_shard)
    parallelize.apply_fsdp(
        model,
        dp_mesh=object(),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
    )

    assert calls[1][0] is model.layers["0"]
    assert calls[1][1]["mp_policy"].cast_forward_inputs is True
    assert calls[2][0] is model.layers["1"]
    assert calls[2][1]["mp_policy"].cast_forward_inputs is False


def test_nemotron_fsdp_preserves_carrier_layer_input_dtypes(monkeypatch):
    model = _NemotronModel()
    model.layers[1]._fsdp_preserve_forward_input_dtypes = True
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(nemotron_h, "fully_shard", fake_fully_shard)
    nemotron_h.apply_nemotron_fsdp(
        model,
        dp_mesh=object(),
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    assert calls[1][0] is model.layers[0]
    assert calls[1][1]["mp_policy"].cast_forward_inputs is True
    assert calls[2][0] is model.layers[1]
    assert calls[2][1]["mp_policy"].cast_forward_inputs is False
