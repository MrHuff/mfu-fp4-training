import pytest
import torch
import torch.nn as nn

from low_bits_training.models.nemotron_h_hf.modeling_nemotron_h import (
    _mamba_gated_out_projection,
)


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.group_size = 1024
        self.variance_epsilon = 1e-5
        self.calls = 0

    def forward(self, scan, gate):
        self.calls += 1
        return scan + gate


class _Projection(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.fused = fused
        self.calls = 0

    def forward(self, value):
        return value * 2

    def forward_mamba_gated(self, scan, gate, weight, epsilon):
        if not self.fused:
            raise AssertionError("disabled fused route fired")
        self.calls += 1
        assert weight.shape == (8,)
        assert epsilon == 1e-5
        return scan - gate


class _PlainProjection(nn.Module):
    def forward(self, value):
        return value * 2


def test_mamba_gated_projection_prefers_capability(monkeypatch):
    monkeypatch.setenv("LBT_NEMOTRON_H_FUSED_MAMBA_GATED_OUT_PROJ", "1")
    monkeypatch.delattr(
        _mamba_gated_out_projection,
        "_fused_route_logged",
        raising=False,
    )
    norm = _Norm()
    projection = _Projection(fused=True)
    scan = torch.ones(2, 8)
    gate = torch.full_like(scan, 0.25)

    output = _mamba_gated_out_projection(projection, norm, scan, gate)

    torch.testing.assert_close(output, scan - gate)
    assert projection.calls == 1
    assert norm.calls == 0
    assert _mamba_gated_out_projection._fused_route_logged


def test_mamba_gated_projection_can_fall_back(monkeypatch):
    monkeypatch.setenv("LBT_NEMOTRON_H_FUSED_MAMBA_GATED_OUT_PROJ", "0")
    norm = _Norm()
    projection = _Projection(fused=False)
    scan = torch.ones(2, 8)
    gate = torch.full_like(scan, 0.25)

    output = _mamba_gated_out_projection(projection, norm, scan, gate)

    torch.testing.assert_close(output, (scan + gate) * 2)
    assert projection.calls == 0
    assert norm.calls == 1


def test_mamba_gated_projection_is_opt_in(monkeypatch):
    monkeypatch.delenv(
        "LBT_NEMOTRON_H_FUSED_MAMBA_GATED_OUT_PROJ",
        raising=False,
    )
    norm = _Norm()
    projection = _Projection(fused=False)
    scan = torch.ones(2, 8)
    gate = torch.full_like(scan, 0.25)

    output = _mamba_gated_out_projection(projection, norm, scan, gate)

    torch.testing.assert_close(output, (scan + gate) * 2)
    assert projection.calls == 0
    assert norm.calls == 1


def test_mamba_gated_projection_required_capability_fails_fast(monkeypatch):
    monkeypatch.setenv("LBT_NEMOTRON_H_FUSED_MAMBA_GATED_OUT_PROJ", "1")
    monkeypatch.setenv(
        "LBT_NEMOTRON_H_REQUIRE_FUSED_MAMBA_GATED_OUT_PROJ",
        "1",
    )
    norm = _Norm()

    with pytest.raises(
        RuntimeError,
        match="requires fused Nemotron-H Mamba gated output projection",
    ):
        _mamba_gated_out_projection(
            _PlainProjection(),
            norm,
            torch.ones(2, 8),
            torch.full((2, 8), 0.25),
        )

    assert norm.calls == 0
