from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.mxfp4_h32 import analyze, launcher, render, route_contract, runner


def _metric_line(step: int, offset: int = 0) -> str:
    return (
        f"step: {step} loss: {2.0 + offset / 10:.4f} "
        f"grad_norm: {0.1 + offset / 100:.4f} "
        f"memory: {100 + offset}GiB(peak) tps: {30000 + offset:,} "
        f"tflops: {2000 + offset:.1f} mfu: {80 + offset / 10:.2f}%"
    )


def test_route_spec_and_environment_are_fail_closed() -> None:
    spec = route_contract.load_spec()
    assert len(spec["environment"]) == 100
    assert len(route_contract.expected_logical_keys()) == 128
    inherited = {
        "PATH": "/usr/bin",
        "MXFP4_UNRELATED_OLD_SELECTOR": "1",
        "NVFP4_OLD_SELECTOR": "1",
    }
    clean = route_contract.scrub_scientific_environment(inherited)
    clean.update(spec["environment"])
    route_contract.validate_environment(spec, clean)
    clean["MXFP4_RHT_WEIGHT"] = "1"
    with pytest.raises(RuntimeError, match="environment drifted"):
        route_contract.validate_environment(spec, clean)


def test_launcher_emits_explicit_standard_multinode_geometry() -> None:
    geometry = launcher.LaunchGeometry(
        nodes=8,
        processes_per_node=4,
        node_rank=3,
        master_addr="coordinator.example",
        master_port=29500,
        rendezvous_id="science-probe",
    )
    command = launcher.torchrun_command(geometry, Path("train.py"), ["--x=1"])
    assert command[:3] == ["torchrun", "--nnodes", "8"]
    assert command[command.index("--node_rank") + 1] == "3"
    assert command[command.index("--rdzv_endpoint") + 1] == "coordinator.example:29500"
    assert command[-2:] == ["train.py", "--x=1"]


def test_runner_installs_live_route_contract_and_compiled_regular_ce(tmp_path: Path) -> None:
    spec = route_contract.load_spec()
    bindings = {
        "repository": tmp_path / "repo",
        "runtime": tmp_path / "runtime",
        "model_assets": tmp_path / "model",
        "dataset": tmp_path / "dataset",
        "output_dir": tmp_path / "output",
    }
    arguments = runner.train_arguments(spec, bindings, node_rank=0)
    assert (
        "--model.converters=bfloat16,mxfp4_tk,"
        "mxfp4_h32_benchmark_contract,fp32_master"
    ) in arguments
    assert "--job.experimental-modules=mxfp4_h32_benchmark_contract" in arguments
    assert "--compile.enable" in arguments
    assert "--compile.components=loss" in arguments
    assert "--training.no-enable-cce" in arguments
    assert "--checkpoint.no-enable" in arguments


def test_analyzer_requires_every_process_and_node(tmp_path: Path) -> None:
    summaries = []
    for node_rank in range(2):
        log = tmp_path / f"node-{node_rank}.log"
        lines = [
            _metric_line(step, node_rank * 2 + local_rank)
            for step in (1, 2)
            for local_rank in range(2)
        ]
        log.write_text("\n".join(lines) + "\n")
        summary = analyze.build_node_summary(
            log,
            run_id="fixture",
            route="mxfp4-h32",
            node_rank=node_rank,
            node_count=2,
            local_processes=2,
            final_step=2,
            steady_start=1,
            steady_end=2,
            source_sha256="a" * 64,
            route_contract_sha256="b" * 64,
            world_size=4,
            local_batch=2,
            gradient_accumulation=2,
            global_batch=16,
        )
        path = tmp_path / f"node-{node_rank}.json"
        path.write_text(json.dumps(summary))
        summaries.append(path)
    aggregate = analyze.aggregate_node_summaries(summaries)
    assert aggregate["complete"] is True
    assert aggregate["world_metric_records_per_step"] == 4
    assert aggregate["steady_state"]["updates"] == 2
    with pytest.raises(ValueError, match="expected 2 node summaries"):
        analyze.aggregate_node_summaries(summaries[:1])


def test_renderer_binds_only_local_inputs_outside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    runtime_root = repository / "fp4_runtime"
    torchtitan = repository / "torchtitan_submodule"
    model = tmp_path / "model"
    dataset = tmp_path / "dataset"
    for path in (runtime_root, torchtitan, model, dataset):
        path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "results"
    plan_path = output / "effective-plan.json"
    spec = route_contract.load_spec()

    def fake_git_head(path: Path) -> str:
        resolved = path.resolve()
        if resolved == runtime_root.resolve():
            return spec["runtime"]["fp4_commit"]
        if resolved == torchtitan.resolve():
            return spec["runtime"]["torchtitan_commit"]
        return "c" * 40

    monkeypatch.setattr(render, "_git_head", fake_git_head)
    plan = render.render_plan(
        spec_path=route_contract.DEFAULT_SPEC,
        model_assets=model,
        dataset=dataset,
        output_dir=output,
        plan_path=plan_path,
        repository=repository,
        runtime=runtime_root,
    )
    assert runner.load_plan(plan_path) == plan
    serialized = plan_path.read_text()
    assert "://" not in serialized
    assert "checkpoint_metadata" not in serialized.lower()
    assert "source_job" not in serialized.lower()


def test_renderer_rejects_output_inside_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    for path in (
        repository,
        repository / "fp4_runtime",
        repository / "torchtitan_submodule",
        tmp_path / "model",
        tmp_path / "dataset",
    ):
        path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="outside the checkout"):
        render.render_plan(
            spec_path=route_contract.DEFAULT_SPEC,
            model_assets=tmp_path / "model",
            dataset=tmp_path / "dataset",
            output_dir=repository / "results",
            plan_path=repository / "results" / "plan.json",
            repository=repository,
            runtime=repository / "fp4_runtime",
        )


def test_public_benchmark_sources_contain_no_recovered_cluster_identity() -> None:
    root = Path(__file__).resolve().parents[2] / "benchmarks" / "mxfp4_h32"
    forbidden = (
        "/" + "workspace" + "/",
        "/" + "volt" + "/",
        "s3" + "://",
        "graphcore",
        "WANDB" + "_API_KEY",
        "AWS_SECRET" + "_ACCESS_KEY",
        "JOB" + "_UID",
        "WORKLOAD" + "_UID",
    )
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8")
            assert not any(value in text for value in forbidden), path


def test_public_replay_sources_contain_no_private_provenance() -> None:
    repository = Path(__file__).resolve().parents[2]
    paths = [
        repository / "low_bits_training/analysis/layerwise_replay_capture.py",
        repository / "tools/capture_llama_layerwise_replay.py",
        repository / "tools/microscope_llama_quantized_operands.py",
        repository / "benchmarks/operand_hybrid_wgrad/README.md",
        repository / "benchmarks/operand_hybrid_wgrad/reference_results.json",
    ]
    forbidden = (
        "/" + "workspace" + "/",
        "/" + "volt" + "/",
        "s3" + "://",
        "source" + "_job_id",
        "WANDB" + "_API_KEY",
        "AWS_SECRET" + "_ACCESS_KEY",
        "AWS_SESSION" + "_TOKEN",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
