from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from low_bits_training.quantization import mixed_fp4_converter as mixed
from low_bits_training.quantization.mxfp4_tk_converter import (
    MXFP4RMSNormLinearTK,
)
from low_bits_training.quantization.nemotron_h_projection_policy import (
    NemotronHFusedAttentionWrapper,
)


MIXED_ENV_KEYS = (
    "LBT_FP4_MIXED_DEFAULT_BACKEND",
    "LBT_FP4_MIXED_ATTENTION_BACKEND",
    "LBT_FP4_MIXED_FFN_BACKEND",
    "LBT_FP4_MIXED_HEAD_BACKEND",
    "LBT_FP4_MIXED_LAYERS",
    "LBT_FP4_MIXED_MAMBA_BACKEND",
    "LBT_FP4_MIXED_POLICY",
    "LBT_FP4_MIXED_SPLIT_LAYER",
    "LBT_FP4_MIXED_TAIL_LAYERS",
    "LBT_LOCALCTA_V4_PROFILE",
    "FP4_ATTN_BACKEND",
    "FP4_FFN_BACKEND",
    "USE_NVFP4_MXFP4_LIVE_PATH",
    "USE_FP4_CODA_EXACT_CDE",
    "USE_FP4_CODA_EXACT_CDE_WO",
    "USE_TK_GEMM",
    "USE_TK_FFN_DISABLE_WGRAD_STREAM",
    "USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV",
    "USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD",
    "USE_TK_FFN_REQUANT_H13_OPERANDS",
    "USE_TK_FFN_SEPARATE_WGRAD_STREAM",
    "USE_TK_LOCALCTA",
    "USE_TK_LOCALCTA_FUSED",
    "USE_TK_LOCALCTA_VARIANT",
    "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT",
    "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND",
    "USE_TK_QKV_LOCALCTA_FAST_ACT",
    "USE_TK_QKV_LOCALCTA_WEIGHT_OVERLAP",
    "USE_TK_QUANT",
    "USE_TK_RMSNORM_BWD_SINGLE_OUT",
)


@pytest.fixture(autouse=True)
def _clear_mixed_environment(monkeypatch):
    for key in MIXED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_tail_mxfp4_routes_exact_final_count(monkeypatch):
    layers = list(range(32))
    monkeypatch.setenv("LBT_FP4_MIXED_POLICY", "tail_mxfp4")
    monkeypatch.setenv("LBT_FP4_MIXED_TAIL_LAYERS", "8")

    routes = {
        idx: mixed._default_backend_for_layer(idx, layers)
        for idx in layers
    }

    assert [idx for idx, backend in routes.items() if backend == "mxfp4"] == list(
        range(24, 32)
    )
    assert all(routes[idx] == "localcta" for idx in range(24))


def test_explicit_routes_override_policy(monkeypatch):
    monkeypatch.setattr(mixed, "_block_layer_indices", lambda model: list(range(4)))
    monkeypatch.setenv("LBT_FP4_MIXED_POLICY", "all_localcta")
    monkeypatch.setenv("LBT_FP4_MIXED_LAYERS", "mxfp4:2,4")

    assert mixed._build_layer_routes(object()) == {
        0: "localcta",
        1: "mxfp4",
        2: "localcta",
        3: "mxfp4",
    }


def test_component_overrides_route_attention_and_ffn_independently(monkeypatch):
    routes = {0: "localcta", 1: "mxfp4"}
    monkeypatch.setenv("LBT_FP4_MIXED_ATTENTION_BACKEND", "mxfp4")
    monkeypatch.setenv("LBT_FP4_MIXED_FFN_BACKEND", "localcta-v4")

    assert mixed._projection_backend(routes, "mxfp4", 0, "attention") == "mxfp4"
    assert mixed._projection_backend(routes, "mxfp4", 1, "ffn") == "localcta"
    assert mixed._projection_backend(routes, "mxfp4", 0, "head") == "mxfp4"


def test_mamba_component_override_falls_back_to_shared_backend(monkeypatch):
    routes = {0: "localcta"}
    monkeypatch.setenv("LBT_FP4_MIXED_MAMBA_BACKEND", "v5")

    assert mixed._projection_backend(routes, "mxfp4", 0, "mamba_in") == "v5"
    assert mixed._projection_backend(routes, "mxfp4", 0, "mamba_out") == "v5"


def test_explicit_v5_routes_can_share_model_with_mxfp4(monkeypatch):
    monkeypatch.setattr(mixed, "_block_layer_indices", lambda model: list(range(4)))
    monkeypatch.setenv("LBT_FP4_MIXED_LAYERS", "v5:1-3;mxfp4:4")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    monkeypatch.setenv(
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT", "1"
    )
    monkeypatch.setenv("USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV", "1")
    monkeypatch.setenv("USE_TK_QKV_LOCALCTA_DGRAD_BACKEND", "split3")
    monkeypatch.setenv("USE_TK_QKV_LOCALCTA_FAST_ACT", "1")
    monkeypatch.setenv("USE_TK_QKV_LOCALCTA_WEIGHT_OVERLAP", "1")
    monkeypatch.setenv("USE_TK_FFN_DISABLE_WGRAD_STREAM", "1")
    monkeypatch.setenv("USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD", "1")
    monkeypatch.setenv("USE_TK_FFN_REQUANT_H13_OPERANDS", "1")
    monkeypatch.setenv("USE_TK_FFN_SEPARATE_WGRAD_STREAM", "1")
    monkeypatch.setenv("USE_TK_RMSNORM_BWD_SINGLE_OUT", "1")
    monkeypatch.setenv("LBT_LOCALCTA_V4_PROFILE", "highwater")
    monkeypatch.setenv("USE_FP4_CODA_EXACT_CDE", "1")
    monkeypatch.setenv("USE_FP4_CODA_EXACT_CDE_WO", "1")

    routes = mixed._build_layer_routes(object())

    assert routes == {0: "v5", 1: "v5", 2: "v5", 3: "mxfp4"}
    assert mixed._configure_regular_nvfp4(routes) == "v5"
    assert mixed.os.environ["USE_TK_LOCALCTA"] == "0"
    assert mixed.os.environ["FP4_ATTN_BACKEND"] == "tk"
    assert mixed.os.environ["FP4_FFN_BACKEND"] == "tk"
    assert "USE_TK_LOCALCTA_VARIANT" not in mixed.os.environ
    assert (
        "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT"
        not in mixed.os.environ
    )
    assert "USE_TK_FFN_LOCALCTA_INPLACE_H13_DERIV" not in mixed.os.environ
    assert "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND" not in mixed.os.environ
    assert "USE_TK_QKV_LOCALCTA_FAST_ACT" not in mixed.os.environ
    assert "USE_TK_QKV_LOCALCTA_WEIGHT_OVERLAP" not in mixed.os.environ
    assert mixed.os.environ["USE_TK_FFN_DISABLE_WGRAD_STREAM"] == "0"
    assert "USE_TK_FFN_RECOMPUTE_H_FOR_W2_WGRAD" not in mixed.os.environ
    assert "USE_TK_FFN_REQUANT_H13_OPERANDS" not in mixed.os.environ
    assert "USE_TK_FFN_SEPARATE_WGRAD_STREAM" not in mixed.os.environ
    assert mixed.os.environ["USE_TK_RMSNORM_BWD_SINGLE_OUT"] == "1"
    assert mixed.os.environ["LBT_LOCALCTA_V4_PROFILE"] == "off"
    assert mixed.os.environ["USE_FP4_CODA_EXACT_CDE"] == "0"
    assert mixed.os.environ["USE_FP4_CODA_EXACT_CDE_WO"] == "0"


def test_localcta_mixed_routes_do_not_clear_exact_cde(monkeypatch):
    monkeypatch.setenv("USE_FP4_CODA_EXACT_CDE", "1")
    monkeypatch.setenv("USE_FP4_CODA_EXACT_CDE_WO", "1")
    monkeypatch.setattr(
        mixed,
        "_configure_regular_localcta_v4",
        lambda: "localcta-v4",
    )

    assert (
        mixed._configure_regular_nvfp4({0: "localcta", 1: "mxfp4"})
        == "localcta-v4"
    )
    assert mixed.os.environ["USE_FP4_CODA_EXACT_CDE"] == "1"
    assert mixed.os.environ["USE_FP4_CODA_EXACT_CDE_WO"] == "1"


def test_mixed_routes_reject_two_nvfp4_kernel_families():
    with pytest.raises(ValueError, match="localCTA v4 and v5 cannot coexist"):
        mixed._configure_regular_nvfp4({0: "localcta", 1: "v5", 2: "mxfp4"})


def test_explicit_routes_reject_nonexistent_layers(monkeypatch):
    monkeypatch.setattr(mixed, "_block_layer_indices", lambda model: list(range(4)))
    monkeypatch.setenv("LBT_FP4_MIXED_LAYERS", "mxfp4:5")

    with pytest.raises(ValueError, match="not present.*5"):
        mixed._build_layer_routes(object())


def test_routes_reject_models_without_transformer_layers(monkeypatch):
    monkeypatch.setattr(mixed, "_block_layer_indices", lambda model: [])

    with pytest.raises(ValueError, match="did not discover"):
        mixed._build_layer_routes(object())


class _FusedAttentionWithoutCDE(nn.Module):
    def forward_wo(self, attn_output, residual=None):
        return attn_output if residual is None else attn_output + residual


def _nemotron_attention_shell(fused):
    original = SimpleNamespace(
        num_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_dropout=0.0,
        layer_idx=0,
    )
    return NemotronHFusedAttentionWrapper(original, fused)


def test_mxfp4_nemotron_attention_does_not_receive_disabled_cde_keyword():
    wrapper = _nemotron_attention_shell(_FusedAttentionWithoutCDE())
    output = torch.ones(2, 4)
    residual = torch.full((2, 4), 2.0)

    result = wrapper._forward_wo(
        output,
        residual=residual,
        cde_emit=False,
    )

    torch.testing.assert_close(result, output + residual)


def test_mxfp4_nemotron_attention_rejects_requested_unsupported_cde():
    wrapper = _nemotron_attention_shell(_FusedAttentionWithoutCDE())

    with pytest.raises(RuntimeError, match="does not support.*CDE"):
        wrapper._forward_wo(
            torch.ones(2, 4),
            residual=None,
            cde_emit=True,
        )


class _MetaRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(4096, device="meta", dtype=torch.bfloat16)
        )
        self.variance_epsilon = 1e-5


class _MetaMambaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_type = "mamba"
        self.norm = _MetaRMSNorm()
        self.mixer = nn.Module()
        self.mixer.in_proj = nn.Linear(
            4096,
            18560,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )


class _MetaNemotron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_MetaMambaBlock()])


def test_mixed_mxfp4_fuses_nemotron_mamba_rms_into_in_projection():
    model = _MetaNemotron()

    counts = mixed._replace_nemotron_h_mamba_rms_in_projections(
        model,
        backend_for_projection=lambda layer, kind, name: "mxfp4",
        tail_bf16_names=set(),
        final_bf16_layer_indices=set(),
    )

    assert counts == {"localcta": 0, "v5": 0, "mxfp4": 1}
    assert isinstance(model.layers[0].mixer.in_proj, MXFP4RMSNormLinearTK)
    assert isinstance(model.layers[0].norm, mixed.MXFP4NormIdentity)
