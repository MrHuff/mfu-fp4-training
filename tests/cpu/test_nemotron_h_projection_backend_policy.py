import torch.nn as nn

from low_bits_training.quantization.nemotron_h_projection_policy import (
    replace_nemotron_h_projection_linears,
)


class _MambaMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(16, 32, bias=False)
        self.out_proj = nn.Linear(16, 16, bias=False)


class _MambaBlock(nn.Module):
    block_type = "mamba"

    def __init__(self):
        super().__init__()
        self.mixer = _MambaMixer()
        self.norm = nn.LayerNorm(16)


class _NemotronModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MambaBlock()])
        self.lm_head = nn.Linear(16, 64, bias=False)


def test_projection_backend_override_receives_projection_kind(monkeypatch):
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ", "1")
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_MAMBA_OUT_PROJ", "1")
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_OUTPUT_HEAD", "1")

    model = _NemotronModel()
    replacements = {}

    def make_linear(linear, name, backend):
        del linear
        replacements[name] = backend
        return nn.Identity()

    counts = replace_nemotron_h_projection_linears(
        model,
        make_linear=make_linear,
        backend_for_layer=lambda layer_idx: "localcta",
        backend_for_projection=(
            lambda layer_idx, kind, name: "mxfp4"
            if kind in {"mamba_in", "mamba_out", "head"}
            else "localcta"
        ),
        label="TEST",
    )

    assert counts == {"attention": 0, "mamba_in": 1, "mamba_out": 1, "head": 1}
    assert replacements == {
        "layers.0.mixer.in_proj": "mxfp4",
        "layers.0.mixer.out_proj": "mxfp4",
        "lm_head": "mxfp4",
    }


def test_projection_backend_defaults_to_layer_route(monkeypatch):
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ", "1")
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_MAMBA_OUT_PROJ", "0")
    monkeypatch.setenv("LBT_NEMOTRON_H_FP4_OUTPUT_HEAD", "0")

    model = _NemotronModel()
    replacements = {}

    def make_linear(linear, name, backend):
        del linear
        replacements[name] = backend
        return nn.Identity()

    replace_nemotron_h_projection_linears(
        model,
        make_linear=make_linear,
        backend_for_layer=lambda layer_idx: "localcta",
        label="TEST",
    )

    assert replacements == {"layers.0.mixer.in_proj": "localcta"}
