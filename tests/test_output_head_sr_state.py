from __future__ import annotations

from pathlib import Path

import pytest
import torch

from low_bits_training.batch_resume import install_checkpoint_aligned_dataloader
from low_bits_training.cce import head_sr_state as sr


def _state(
    step_ref: list[int],
    *,
    rank: int = 0,
    world_size: int = 1,
    ga: int = 2,
) -> sr.OutputHeadSRState:
    return sr.OutputHeadSRState(
        device="cpu",
        user_seed=42,
        user_subsequence_base=17,
        training_steps=100,
        gradient_accumulation_steps=ga,
        step_getter=lambda: step_ref[0],
        reservation_margin=4,
        rank=rank,
        world_size=world_size,
    )


def _advance(state: sr.OutputHeadSRState, count: int) -> None:
    state.get()[1].add_(count * sr.SUBSEQUENCE_STRIDE)


def test_state_roundtrip_preserves_exact_next_microbatch_stream():
    step = [0]
    original = _state(step)
    _advance(original, 2)
    step[0] = 1
    checkpoint = original.state_dict()

    resumed = _state(step)
    resumed.load_state_dict(checkpoint)
    resumed.validate_progress()

    assert torch.equal(resumed.get(), original.get())
    _advance(original, 1)
    _advance(resumed, 1)
    assert torch.equal(resumed.get(), original.get())


def test_dcp_roundtrip_and_metadata_schema(tmp_path):
    from torch.distributed.checkpoint import load, save

    step = [0]
    original = _state(step)
    _advance(original, 2)
    step[0] = 1
    checkpoint_id = str(tmp_path / "step-1")
    save({sr.CHECKPOINT_KEY: original}, checkpoint_id=checkpoint_id)
    assert sr.checkpoint_output_head_sr_schema(checkpoint_id) == "v1"

    resumed_step = [0]
    resumed = _state(resumed_step)
    load({sr.CHECKPOINT_KEY: resumed}, checkpoint_id=checkpoint_id)
    resumed_step[0] = 1
    resumed.validate_progress()
    assert torch.equal(resumed.get(), original.get())


def test_progress_fails_closed_on_extra_or_missing_invocation():
    step = [1]
    state = _state(step)
    _advance(state, 1)
    with pytest.raises(RuntimeError, match="step/microbatch geometry"):
        state.validate_progress()


def test_rank_seed_namespace_is_collision_free():
    step = [0]
    rank0 = _state(step, rank=0, world_size=2)
    rank1 = _state(step, rank=1, world_size=2)
    assert int(rank0.get()[0]) != int(rank1.get()[0])
    assert int(rank0.get()[1]) == int(rank1.get()[1]) == 17


def test_load_rejects_version_world_and_ga_drift():
    step = [0]
    checkpoint = _state(step).state_dict()

    bad_version = dict(checkpoint)
    bad_version["version"] = torch.tensor([99], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="unsupported output-head SR state version"):
        _state(step).load_state_dict(bad_version)

    bad_world = dict(checkpoint)
    bad_world["world_size"] = torch.tensor([2], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="world_size differs"):
        _state(step).load_state_dict(bad_world)

    with pytest.raises(RuntimeError, match="gradient-accumulation geometry"):
        _state(step, ga=1).load_state_dict(checkpoint)


def test_supported_route_rejects_inert_column_sr_flags(monkeypatch):
    for name in (
        "FP4_CCE_V4_NVFP4_G_ROW_DATA_SR",
        "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW",
        "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE",
        "FP4_CCE_V4_MIXED_DW_MXFP8_COLS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_X_COL_DATA_SR", "1")
    with pytest.raises(RuntimeError, match="column flags are inert"):
        sr._validate_supported_route()

    monkeypatch.setenv("FP4_CCE_V4_NVFP4_X_COL_DATA_SR", "0")
    monkeypatch.setenv("FP4_CCE_V4_NVFP4_G_COL_DATA_SR", "0")
    sr._validate_supported_route()


class _FakeCheckpointer:
    def __init__(self, step_ref: list[int]) -> None:
        self.states = {}
        self.ft_manager = None
        self.step_ref = step_ref
        self.loads = 0

    def dcp_load(
        self,
        state_dict,
        checkpoint_id,
        from_hf,
        from_quantized,
    ):
        self.loads += 1
        assert sr.CHECKPOINT_KEY not in state_dict
        self.step_ref[0] = 33_000
        return "loaded"


class _FakeLogger:
    def warning(self, *_args, **_kwargs) -> None:
        return None


def test_missing_legacy_state_requires_explicit_new_phase(monkeypatch):
    step = [0]
    state = _state(step)
    checkpointer = _FakeCheckpointer(step)
    sr.register_with_checkpointer(checkpointer, state, _FakeLogger())
    monkeypatch.setattr(sr, "checkpoint_output_head_sr_schema", lambda _path: "missing")
    payload = {sr.CHECKPOINT_KEY: state}

    with pytest.raises(RuntimeError, match="rejected by default"):
        checkpointer.dcp_load(payload, "/legacy", False, False)
    assert checkpointer.loads == 0

    monkeypatch.setenv(sr.MISSING_POLICY_ENV, sr.START_NEW_PHASE_POLICY)
    assert checkpointer.dcp_load(payload, "/legacy", False, False) == "loaded"
    assert state.phase_origin_step == 33_000
    assert int(state.get()[1]) == 17
    state.validate_progress()


class _CoexistDataloader:
    def state_dict(self):
        return {"cursor": 7}

    def load_state_dict(self, state_dict):
        assert state_dict == {"cursor": 7}


class _CoexistCheckpointer(_FakeCheckpointer):
    def __init__(self, step_ref, dataloader):
        super().__init__(step_ref)
        self.states = {"dataloader": dataloader}
        self.ft_states = {"dataloader": dataloader}

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        assert set(state_dict) == {"dataloader"}
        assert state_dict["dataloader"] is self.states["dataloader"]
        return super().dcp_load({}, checkpoint_id, from_hf, from_quantized)


def test_batch_adapter_and_head_sr_state_coexist_on_legacy_load(monkeypatch):
    step = [0]
    dataloader = _CoexistDataloader()
    checkpointer = _CoexistCheckpointer(step, dataloader)
    adapter = install_checkpoint_aligned_dataloader(checkpointer, dataloader)
    state = _state(step)
    sr.register_with_checkpointer(checkpointer, state, _FakeLogger())

    assert checkpointer.states == {
        "dataloader": adapter,
        sr.CHECKPOINT_KEY: state,
    }
    assert checkpointer.ft_states == {"dataloader": adapter}

    monkeypatch.setattr(sr, "checkpoint_output_head_sr_schema", lambda _path: "missing")
    monkeypatch.setenv(sr.MISSING_POLICY_ENV, sr.START_NEW_PHASE_POLICY)
    assert (
        checkpointer.dcp_load(dict(checkpointer.states), "/legacy", False, False)
        == "loaded"
    )
    assert state.phase_origin_step == 33_000
    state.validate_progress()


def test_trainer_installs_batch_and_head_state_before_compile():
    trainer = Path(__file__).parents[1] / "low_bits_training" / "trainer.py"
    source = trainer.read_text()
    localcta_install = source.index("build_localcta_sr_state_for_trainer")
    batch_install = source.index("self._prefetch_checkpoint_dataloader =")
    head_install = source.index("build_output_head_sr_state_for_trainer")
    compile_gate = source.index("# Manual Torch.Compile")
    assert localcta_install < batch_install < head_install < compile_gate
