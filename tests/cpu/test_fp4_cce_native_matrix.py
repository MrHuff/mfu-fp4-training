import importlib.util
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_SCRIPT = REPO_ROOT / "tools" / "run_fp4_cce_train_matrix.py"
MATRIX_CONFIG = (
    REPO_ROOT
    / "train_configs"
    / "ablations"
    / "fp4_cce_slimpajama"
    / "8b"
    / "final_layer_cce"
    / "nvfp4_v4_pcache_matrix.toml"
)


def _load_matrix_script():
    spec = importlib.util.spec_from_file_location(
        "run_fp4_cce_train_matrix",
        MATRIX_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_matrix_is_bf16_and_v4_pcache_cells():
    module = _load_matrix_script()
    variants = module.DEFAULT_VARIANTS

    assert [variant.label for variant in variants] == [
        "native-bf16-control",
        "nv-v4-pcache",
        "mx-v4-pcache",
    ]

    control_args = " ".join(variants[0].cli_args)
    assert "--fp4_cce.backend=native_mxfp4" in control_args
    assert "--fp4_cce.forward_precision=bf16" in control_args
    assert "--fp4_cce.backward_precision=bf16" in control_args

    for variant, backend in zip(variants[1:], ("nvfp4", "mxfp4"), strict=True):
        args = " ".join(variant.cli_args)
        assert f"--fp4_cce.backend={backend}" in args
        assert "--fp4_cce.implementation=v4" in args
        assert "--fp4_cce.quant_mode=enc" in args
        assert "triton" not in args.lower()
        assert "true_nuclear" not in args.lower()
        assert dict(variant.env)["FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE"] == "1"

    assert dict(variants[1].env)["FP4_CCE_NVFP4_EXACT_NORM_QUANT"] == "0"
    assert dict(variants[1].env)["FP4_CCE_V4_NVFP4_G_CACHE"] == "0"

    assert (
        dict(module.VARIANT_BY_LABEL["nv-v4-gcache-unit-bound"].env)[
            "FP4_CCE_V4_NVFP4_G_CACHE"
        ]
        == "1"
    )
    p_sr_env = dict(module.VARIANT_BY_LABEL["nv-v4-pcache-dynamic-p-sr"].env)
    assert p_sr_env["FP4_CCE_V4_NVFP4_G_CACHE"] == "0"
    assert p_sr_env["FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE"] == "0"
    assert p_sr_env["FP4_CCE_V4_NVFP4_P_DATA_SR"] == "1"
    target_split_env = dict(
        module.VARIANT_BY_LABEL["nv-v4-pcache-target-split-sr"].env
    )
    assert target_split_env["FP4_CCE_V4_NVFP4_G_CACHE"] == "0"
    assert target_split_env["FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE"] == "0"
    assert target_split_env["FP4_CCE_V4_NVFP4_P_TARGET_SPLIT"] == "1"
    assert target_split_env["FP4_CCE_V4_NVFP4_P_DATA_SR"] == "1"
    assert target_split_env["FP4_CCE_V4_STRICT_FUSED_SPARSE"] == "1"
    target_top1_env = dict(
        module.VARIANT_BY_LABEL["nv-v4-pcache-target-top1-split-sr"].env
    )
    assert target_top1_env["FP4_CCE_V4_NVFP4_P_TARGET_SPLIT"] == "1"
    assert target_top1_env["FP4_CCE_V4_NVFP4_P_TOP1_SPLIT"] == "1"
    assert target_top1_env["FP4_CCE_V4_NVFP4_P_DATA_SR"] == "1"

    for label in (
        "native-bf16-fwd-bf16-bwd",
        "native-fp4-fwd-bf16-bwd",
        "native-bf16-fwd-fp4-bwd",
        "native-fp4-fwd-fp4-bwd",
    ):
        assert label in module.VARIANT_BY_LABEL


def test_default_matrix_config_is_bf16_8b_and_1000_steps():
    module = _load_matrix_script()
    with MATRIX_CONFIG.open("rb") as handle:
        config = tomllib.load(handle)

    assert module.DEFAULT_CONFIG == MATRIX_CONFIG
    assert config["model"]["flavor"] == "8B"
    assert config["model"]["converters"] == ["bfloat16"]
    assert config["training"]["dtype"] == "bfloat16"
    assert config["training"]["steps"] == 1000
    assert config["debug"]["seed"] == 42
    assert config["fp4_cce"] == {
        "enabled": True,
        "backend": "nvfp4",
        "implementation": "v4",
        "quant_mode": "enc",
        "ignore_index": -100,
        "filter_eps": 0.0,
    }


def test_common_bf16_eval_is_enabled_every_50_steps_by_default(monkeypatch):
    module = _load_matrix_script()
    monkeypatch.setattr(sys, "argv", [str(MATRIX_SCRIPT)])
    args = module.parse_args()

    assert args.common_eval is True
    assert args.common_eval_every == 50
