from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("paper_recipe", TOOLS / "paper_recipe.py")
assert SPEC is not None and SPEC.loader is not None
RECIPE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECIPE
SPEC.loader.exec_module(RECIPE)


def _hashed(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _inputs(tmp_path: Path, *, checkpoint_route: str | None = None) -> Path:
    assets = tmp_path / "assets"
    assets.mkdir()
    dataset = tmp_path / "manifest.json"
    cursor = tmp_path / "cursor.json"
    for name, content in (
        ("tokenizer.json", b"tokenizer"),
        ("tokenizer_config.json", b"{}"),
        ("special_tokens_map.json", b"{}"),
    ):
        (assets / name).write_bytes(content)
    dataset.write_text('{"format":"lbt_packed_tokens_manifest_v1","shards":[]}')
    cursor.write_text('{"schema_version":1}')
    value: dict[str, object] = {
        "schema_version": 1,
        "tokenizer": {
            "directory": str(assets),
            "files": {
                name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
            },
        },
        "dataset": {"manifest": _hashed(dataset), "cursor_contract": _hashed(cursor)},
        "output": {"directory": str(tmp_path / "output")},
        "authentication": {"mode": "none"},
    }
    if checkpoint_route is not None:
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        checkpoint_manifest = tmp_path / "checkpoint-manifest.json"
        checkpoint_manifest.write_text('{"schema_version":1}')
        value["checkpoint"] = {
            "directory": str(checkpoint),
            "manifest": _hashed(checkpoint_manifest),
            "route_id": checkpoint_route,
            "step": 1000,
        }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(value))
    return path


def _args(inputs: Path, **overrides) -> argparse.Namespace:
    values = {
        "route": "mxfp4-v4-row-sr-h32-rht-long",
        "inputs": str(inputs),
        "runtime_root": str(ROOT / "fp4_runtime"),
        "nnodes": 8,
        "nproc_per_node": 8,
        "node_rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "resume": False,
        "execute": False,
        "wandb_mode": "disabled",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_h32_plan_is_path_bound_and_route_complete(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setenv("MXFP4_RHT_BLOCK_SIZE", "contaminated")
    command, environment, summary = RECIPE.build_plan(_args(inputs))

    assert summary["world_size"] == 64
    assert summary["global_batch_size"] == 512
    assert summary["external_values_redacted"] is True
    assert environment["MXFP4_RHT_BLOCK_SIZE"] == "32"
    assert environment["MXFP4_RHT_RANDOM_SIGN_MASK"] == "1"
    assert environment["MXFP4_USE_LOCALCTA_DGRAD"] == "0"
    assert environment["LBT_REQUIRE_FRESH_START"] == "1"
    assert any(value.startswith("--training.dataset-path=") for value in command)
    assert "--model.converters=bfloat16,mxfp4_tk,paper_route_contract_v1,paper_regular_bf16_head_v1,fp32_master" in command


def test_resume_requires_matching_external_route(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, checkpoint_route="pure-v5-fused-v1-long")
    args = _args(inputs, resume=True)
    try:
        RECIPE.build_plan(args)
    except ValueError as error:
        assert "checkpoint route_id" in str(error)
    else:
        raise AssertionError("mismatched checkpoint route was accepted")


def test_topology_change_requires_new_route_identity(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    args = _args(inputs, nnodes=4, nproc_per_node=8)
    try:
        RECIPE.build_plan(args)
    except ValueError as error:
        assert "requires world size 64" in str(error)
    else:
        raise AssertionError("topology drift was accepted")


def test_every_executable_route_environment_is_public_and_resolvable() -> None:
    document = json.loads((ROOT / "configs/paper/route_execution.json").read_text())
    for route in document["routes"]:
        if route["execution_status"] == "withheld_invalid":
            assert route["environment"] is None
            assert not route["converters"]
            continue
        path = ROOT / route["environment"]
        environment = RECIPE._route_environment(path)
        assert environment
        assert not any("SECRET" in key or "TOKEN" in key or "CREDENTIAL" in key for key in environment)
        private_workspace_prefix = "/" + "workspace/"
        assert not any(
            "://" in value or private_workspace_prefix in value
            for value in environment.values()
        )
        assert not any(key.startswith("WANDB_") for key in environment)
        assert environment["USE_FP4_CONVERT_OUTPUT_HEAD"] == "0"


def test_execution_routes_cover_recorded_contracts() -> None:
    recorded = json.loads(
        (ROOT / "configs/paper/llama8b_160b_recipe_contracts.json").read_text()
    )
    execution = json.loads((ROOT / "configs/paper/route_execution.json").read_text())
    recorded_ids = {route["id"] for route in recorded["routes"]}
    execution_ids = {route["id"] for route in execution["routes"]}
    assert recorded_ids == execution_ids
