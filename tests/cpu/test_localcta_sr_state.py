from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
from low_bits_training.quantization import localcta_sr_state as sr


def _state(keys=("layer.0:sr:a", "layer.0:sr:b"), **overrides):
    kwargs = {
        "device": "cpu",
        "user_seed": 42,
        "user_subsequence_base": 17,
        "training_steps": 71_526,
        "gradient_accumulation_steps": 2,
        "reservation_margin": 4096,
    }
    kwargs.update(overrides)
    return sr.LocalCTASRState(keys, **kwargs)


def _all_rank_checkpoint(*states):
    assert states
    table = torch.stack([state._local_state_matrix().detach().cpu() for state in states])
    return states[0]._checkpoint_state_dict(table)


def _v1_checkpoint(keys=("layer.0:sr:a", "layer.0:sr:b"), *, seed=42):
    keys = tuple(sorted(keys))
    states = torch.tensor(
        [
            [
                sr._as_signed_int64((seed + slot + 1) & sr.UINT64_MAX),
                17 + (slot + 3) * sr.SUBSEQUENCE_STRIDE,
            ]
            for slot in range(len(keys))
        ],
        dtype=torch.int64,
    )
    return {
        "version": torch.tensor([sr.LEGACY_STATE_VERSION], dtype=torch.int64),
        "subsequence_stride": torch.tensor([sr.SUBSEQUENCE_STRIDE], dtype=torch.int64),
        "logical_keys": list(keys),
        "states": states,
    }


def test_seed_namespace_and_user_subsequence_base_are_exact() -> None:
    state = _state(keys=("z", "a"))

    # Keys are sorted: slot 0 is a, slot 1 is z.  Seed addition modulo 2**64
    # is collision-free over this finite slot set; base is unchanged per slot.
    assert state.get("a").tolist() == [43, 17]
    assert state.get("z").tolist() == [44, 17]
    assert state.reservations_per_slot == (71_526 + 1) * 2 + 4096
    span = state.reservations_per_slot * sr.SUBSEQUENCE_STRIDE
    assert (1 << 62) + span <= sr.UINT64_MAX


def test_seed_namespace_is_unique_across_ranks_and_producers() -> None:
    states = [_state(keys=("z", "a"), rank=rank, world_size=4) for rank in range(4)]
    seeds = [state.get(key)[0].item() for state in states for key in state.logical_keys]

    assert len(seeds) == len(set(seeds))
    # Rank zero deliberately retains the v1 namespace.
    assert states[0].get("a")[0].item() == 43
    assert states[0].get("z")[0].item() == 44
    assert states[3].get("a")[0].item() == 49
    assert states[3].get("z")[0].item() == 50


@pytest.mark.parametrize(
    ("rank", "world_size", "error"),
    [(-1, 2, "rank must"), (2, 2, "rank must"), (0, 0, "world_size")],
)
def test_invalid_rank_world_namespace_is_rejected(rank, world_size, error) -> None:
    with pytest.raises(ValueError, match=error):
        _state(rank=rank, world_size=world_size)


def test_uint64_seed_wrap_is_preserved_in_signed_tensor_storage() -> None:
    state = _state(keys=("a", "b"), user_seed=sr.UINT64_MAX)

    assert state.get("a").tolist() == [0, 17]
    assert state.get("b").tolist() == [1, 17]


def test_headroom_validation_rejects_subsequence_overflow() -> None:
    with pytest.raises(ValueError, match="insufficient uint64 headroom"):
        _state(user_subsequence_base=sr.UINT64_MAX)


def test_checkpoint_roundtrip_is_strict_and_snapshot_is_immutable() -> None:
    original = _state()
    original.get("layer.0:sr:a")[1] += 3 * sr.SUBSEQUENCE_STRIDE
    checkpoint = original.state_dict()
    original.get("layer.0:sr:a")[1] += sr.SUBSEQUENCE_STRIDE

    restored = _state(user_seed=999, user_subsequence_base=999)
    restored.load_state_dict(checkpoint)
    assert restored.get("layer.0:sr:a")[1].item() == (17 + 3 * sr.SUBSEQUENCE_STRIDE)
    assert checkpoint["states"][0, 0, 1].item() == (17 + 3 * sr.SUBSEQUENCE_STRIDE)
    assert restored.get("layer.0:sr:a")[0].item() == 43

    incompatible = _state(keys=("layer.0:sr:a", "different"))
    with pytest.raises(RuntimeError, match="manifest differs"):
        incompatible.load_state_dict(checkpoint)


def test_exact_same_rank_resume_restores_seed_and_counter_table() -> None:
    rank0 = _state(rank=0, world_size=2)
    rank1 = _state(rank=1, world_size=2)
    rank0.get("layer.0:sr:a")[1] += sr.SUBSEQUENCE_STRIDE
    rank1.get("layer.0:sr:a")[1] += 7 * sr.SUBSEQUENCE_STRIDE
    rank1.get("layer.0:sr:b")[1] += 9 * sr.SUBSEQUENCE_STRIDE
    checkpoint = _all_rank_checkpoint(rank0, rank1)

    restored = _state(
        rank=1,
        world_size=2,
        user_seed=999,
        user_subsequence_base=999,
    )
    restored.load_state_dict(checkpoint)

    assert torch.equal(restored._local_state_matrix(), rank1._local_state_matrix())
    # The checkpoint seed is active for continuation; the runtime seed remains
    # the base for an explicit reset only.
    restored.reset_to_configured_base()
    assert restored.get("layer.0:sr:a").tolist() == [1002, 999]


def test_independent_same_rank_twins_are_bitwise_deterministic() -> None:
    left = _state(rank=1, world_size=2)
    right = _state(rank=1, world_size=2)
    for state in (left, right):
        state.get("layer.0:sr:a")[1] += 11 * sr.SUBSEQUENCE_STRIDE
        state.get("layer.0:sr:b")[1] += 13 * sr.SUBSEQUENCE_STRIDE

    assert torch.equal(left._local_state_matrix(), right._local_state_matrix())


def test_v2_resume_fails_closed_on_world_rank_or_seed_mismatch() -> None:
    checkpoint = _all_rank_checkpoint(
        _state(rank=0, world_size=2),
        _state(rank=1, world_size=2),
    )

    with pytest.raises(RuntimeError, match="world_size differs"):
        _state(rank=0, world_size=1).load_state_dict(checkpoint)

    corrupt_rank_ids = dict(checkpoint)
    corrupt_rank_ids["rank_ids"] = torch.tensor([1, 0], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="rank namespace is malformed"):
        _state(rank=0, world_size=2).load_state_dict(corrupt_rank_ids)

    corrupt_seed = dict(checkpoint)
    corrupt_seed["states"] = checkpoint["states"].clone()
    corrupt_seed["states"][1, 0, 0] += 1
    with pytest.raises(RuntimeError, match="seed namespace is inconsistent"):
        _state(rank=0, world_size=2).load_state_dict(corrupt_seed)

    corrupt_dtype = dict(checkpoint)
    corrupt_dtype["states"] = checkpoint["states"].to(torch.float64)
    with pytest.raises(RuntimeError, match="must be an int64 tensor"):
        _state(rank=0, world_size=2).load_state_dict(corrupt_dtype)


def test_multi_rank_checkpoint_requires_live_process_group() -> None:
    with pytest.raises(RuntimeError, match="initialized distributed process group"):
        _state(rank=1, world_size=2).state_dict()


def test_explicit_v1_migration_preserves_counters_and_decorrelates_ranks() -> None:
    checkpoint = _v1_checkpoint()
    rank0 = _state(rank=0, world_size=2, user_seed=999)
    rank1 = _state(rank=1, world_size=2, user_seed=999)

    rank0.migrate_from_v1_state_dict(checkpoint)
    rank1.migrate_from_v1_state_dict(checkpoint)

    assert rank0.get("layer.0:sr:a").tolist() == checkpoint["states"][0].tolist()
    assert rank1.get("layer.0:sr:a")[0].item() == 45
    assert rank1.get("layer.0:sr:b")[0].item() == 46
    assert torch.equal(rank1._local_state_matrix()[:, 1], checkpoint["states"][:, 1])

    malformed = dict(checkpoint)
    malformed["states"] = checkpoint["states"].clone()
    malformed["states"][1, 0] += 1
    with pytest.raises(RuntimeError, match="v1 seed slots are inconsistent"):
        rank1.migrate_from_v1_state_dict(malformed)


def test_torch_distributed_checkpoint_stateful_roundtrip(tmp_path) -> None:
    import torch.distributed.checkpoint as dcp

    original = _state()
    original.get("layer.0:sr:b")[1] += 5 * sr.SUBSEQUENCE_STRIDE
    checkpoint_id = str(tmp_path / "step-1")
    dcp.save({sr.CHECKPOINT_KEY: original}, checkpoint_id=checkpoint_id)

    restored = _state(user_seed=999, user_subsequence_base=999)
    dcp.load({sr.CHECKPOINT_KEY: restored}, checkpoint_id=checkpoint_id)
    assert restored.get("layer.0:sr:a").tolist() == original.get("layer.0:sr:a").tolist()
    assert restored.get("layer.0:sr:b").tolist() == original.get("layer.0:sr:b").tolist()
    assert sr.checkpoint_contains_localcta_sr_state(checkpoint_id)
    assert sr.checkpoint_localcta_sr_state_schema(checkpoint_id) == "v2"


def test_checkpoint_schema_classifier_distinguishes_missing_v1_and_unknown(
    tmp_path,
) -> None:
    import torch.distributed.checkpoint as dcp

    missing_id = str(tmp_path / "missing")
    dcp.save({"other": torch.ones(1)}, checkpoint_id=missing_id)
    assert sr.checkpoint_localcta_sr_state_schema(missing_id) == "missing"

    v1_id = str(tmp_path / "v1")
    dcp.save({sr.CHECKPOINT_KEY: _v1_checkpoint()}, checkpoint_id=v1_id)
    assert sr.checkpoint_localcta_sr_state_schema(v1_id) == "v1"

    unknown_id = str(tmp_path / "unknown")
    malformed = _v1_checkpoint()
    malformed["unexpected"] = torch.ones(1)
    dcp.save({sr.CHECKPOINT_KEY: malformed}, checkpoint_id=unknown_id)
    assert sr.checkpoint_localcta_sr_state_schema(unknown_id) == "unknown"


def test_capture_preservation_discards_only_synthetic_reservations() -> None:
    state = _state(keys=("a",))
    before = state.get("a").clone()

    with state.preserve_during_cuda_graph_capture(("a",)):
        state.get("a")[1] += 4 * sr.SUBSEQUENCE_STRIDE

    assert torch.equal(state.get("a"), before)
    state.get("a")[1] += sr.SUBSEQUENCE_STRIDE
    assert state.get("a")[1].item() == 17 + sr.SUBSEQUENCE_STRIDE


def test_stable_model_discovery_and_duplicate_rejection() -> None:
    FusedFFN = type("FusedFeedForwardFP4_TK", (torch.nn.Module,), {})
    FusedAttention = type("FusedAttentionFP4_TK", (torch.nn.Module,), {})
    root = torch.nn.Module()
    root.ffn = FusedFFN()
    root.ffn._lbt_debug_name = "layers.0.feed_forward"
    root.attn = FusedAttention()
    root.attn._lbt_debug_name = "layers.0.attention"

    assert sr.discover_logical_keys((root,)) == tuple(
        sorted(
            (
                sr.ffn_w2_grad_key("layers.0.feed_forward"),
                sr.ffn_deriv_grad_key("layers.0.feed_forward"),
                sr.qkv_grad_key("layers.0.attention:qkv"),
                sr.wo_grad_key("layers.0.attention:wo"),
            )
        )
    )

    root.other_ffn = FusedFFN()
    root.other_ffn._lbt_debug_name = "layers.0.feed_forward"
    with pytest.raises(RuntimeError, match="duplicate localCTA SR logical identity"):
        sr.discover_logical_keys((root,))


class _Logger:
    def __init__(self):
        self.messages = []

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


class _Checkpointer:
    def __init__(self):
        self.states = {"other": object()}
        self.ft_manager = None
        self.loaded = None
        self.legacy_payload = None

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        self.loaded = (state_dict, checkpoint_id, from_hf, from_quantized)
        if self.legacy_payload is not None:
            state_dict[sr.CHECKPOINT_KEY].load_state_dict(self.legacy_payload)
        return "loaded"


def test_old_checkpoint_starts_explicit_new_phase_without_step_inference(
    monkeypatch,
) -> None:
    state = _state()
    checkpointer = _Checkpointer()
    logger = _Logger()
    monkeypatch.setattr(sr, "checkpoint_localcta_sr_state_schema", lambda _: "missing")
    sr.register_with_checkpointer(checkpointer, state, logger)
    state.get("layer.0:sr:a")[1] += 7 * sr.SUBSEQUENCE_STRIDE

    requested = {"other": object(), sr.CHECKPOINT_KEY: state}
    assert checkpointer.dcp_load(requested, "/old", False, False) == "loaded"
    assert sr.CHECKPOINT_KEY not in checkpointer.loaded[0]
    assert sr.CHECKPOINT_KEY in requested
    assert state.get("layer.0:sr:a").tolist() == [43, 17]
    assert state.get("layer.0:sr:b").tolist() == [44, 17]
    assert "no step-based subsequence inference" in logger.messages[0]


def test_new_checkpoint_keeps_sr_state_in_load_request(monkeypatch) -> None:
    state = _state()
    checkpointer = _Checkpointer()
    monkeypatch.setattr(sr, "checkpoint_localcta_sr_state_schema", lambda _: "v2")
    sr.register_with_checkpointer(checkpointer, state, _Logger())

    requested = {"other": object(), sr.CHECKPOINT_KEY: state}
    checkpointer.dcp_load(requested, "/new", False, False)
    assert checkpointer.loaded[0][sr.CHECKPOINT_KEY] is state


def test_v1_resume_is_rejected_without_explicit_migration(monkeypatch) -> None:
    state = _state(rank=1, world_size=2)
    checkpointer = _Checkpointer()
    monkeypatch.setattr(sr, "checkpoint_localcta_sr_state_schema", lambda _: "v1")
    monkeypatch.delenv(sr.V1_MIGRATION_ENV, raising=False)
    sr.register_with_checkpointer(checkpointer, state, _Logger())

    with pytest.raises(RuntimeError, match="rejected by default"):
        checkpointer.dcp_load(
            {"other": object(), sr.CHECKPOINT_KEY: state},
            "/step-30002",
            False,
            False,
        )
    assert checkpointer.loaded is None


def test_v1_resume_requires_named_migration_and_preserves_counter(monkeypatch) -> None:
    state = _state(rank=1, world_size=2, user_seed=999)
    checkpointer = _Checkpointer()
    checkpointer.legacy_payload = _v1_checkpoint()
    logger = _Logger()
    monkeypatch.setattr(sr, "checkpoint_localcta_sr_state_schema", lambda _: "v1")
    monkeypatch.setenv(sr.V1_MIGRATION_ENV, sr.V1_RANK_NAMESPACE_MIGRATION)
    monkeypatch.setenv(sr.V1_EXPECTED_WORLD_SIZE_ENV, "2")
    sr.register_with_checkpointer(checkpointer, state, logger)

    requested = {"other": object(), sr.CHECKPOINT_KEY: state}
    assert checkpointer.dcp_load(requested, "/step-30002", False, False) == "loaded"
    assert checkpointer.loaded[0][sr.CHECKPOINT_KEY] is not state
    assert state.get("layer.0:sr:a").tolist() == [45, 17 + 3 * sr.SUBSEQUENCE_STRIDE]
    assert "not bitwise-continuous" in logger.messages[0]
    assert "no step-based phase inference" in logger.messages[0]


def test_v1_migration_fails_closed_without_verified_matching_world(monkeypatch) -> None:
    for expected_world, error in (
        (None, "did not checkpoint world_size"),
        ("not-an-int", "expected a positive integer"),
        ("1", "differs from the live runtime"),
    ):
        state = _state(rank=1, world_size=2)
        checkpointer = _Checkpointer()
        monkeypatch.setattr(sr, "checkpoint_localcta_sr_state_schema", lambda _: "v1")
        monkeypatch.setenv(sr.V1_MIGRATION_ENV, sr.V1_RANK_NAMESPACE_MIGRATION)
        if expected_world is None:
            monkeypatch.delenv(sr.V1_EXPECTED_WORLD_SIZE_ENV, raising=False)
        else:
            monkeypatch.setenv(sr.V1_EXPECTED_WORLD_SIZE_ENV, expected_world)
        sr.register_with_checkpointer(checkpointer, state, _Logger())

        with pytest.raises(RuntimeError, match=error):
            checkpointer.dcp_load(
                {"other": object(), sr.CHECKPOINT_KEY: state},
                "/step-30002",
                False,
                False,
            )
        assert checkpointer.loaded is None


def test_actual_dcp_v1_loader_migrates_only_after_explicit_opt_in(
    tmp_path, monkeypatch
) -> None:
    import torch.distributed.checkpoint as dcp

    checkpoint_id = str(tmp_path / "step-30002")
    dcp.save({sr.CHECKPOINT_KEY: _v1_checkpoint()}, checkpoint_id=checkpoint_id)

    class DCPCheckpointer(_Checkpointer):
        def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
            self.loaded = (state_dict, checkpoint_id, from_hf, from_quantized)
            return dcp.load(state_dict, checkpoint_id=checkpoint_id)

    state = _state(user_seed=999, user_subsequence_base=999)
    checkpointer = DCPCheckpointer()
    monkeypatch.setenv(sr.V1_MIGRATION_ENV, sr.V1_RANK_NAMESPACE_MIGRATION)
    monkeypatch.setenv(sr.V1_EXPECTED_WORLD_SIZE_ENV, "1")
    sr.register_with_checkpointer(checkpointer, state, _Logger())

    checkpointer.dcp_load(
        {sr.CHECKPOINT_KEY: state},
        checkpoint_id,
        False,
        False,
    )
    assert torch.equal(state._local_state_matrix(), _v1_checkpoint()["states"])


def test_unknown_sr_checkpoint_schema_is_rejected(monkeypatch) -> None:
    state = _state()
    checkpointer = _Checkpointer()
    monkeypatch.setattr(sr, "checkpoint_localcta_sr_state_schema", lambda _: "unknown")
    sr.register_with_checkpointer(checkpointer, state, _Logger())

    with pytest.raises(RuntimeError, match="unrecognized localCTA SR"):
        checkpointer.dcp_load(
            {"other": object(), sr.CHECKPOINT_KEY: state},
            "/malformed",
            False,
            False,
        )


def test_torchft_is_rejected_until_its_fixed_state_closure_includes_sr() -> None:
    checkpointer = _Checkpointer()
    checkpointer.ft_manager = SimpleNamespace()

    with pytest.raises(RuntimeError, match="not yet compatible with TorchFT"):
        sr.register_with_checkpointer(checkpointer, _state(), _Logger())


def test_grad_sr_enable_policy(monkeypatch) -> None:
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_SR_ACTIVATION", "0")
    assert sr.localcta_v4_grad_sr_enabled()

    monkeypatch.setenv("NVFP4_SR_GRAD", "0")
    monkeypatch.setenv("NVFP4_SCALE_SR_GRAD", "0")
    assert not sr.localcta_v4_grad_sr_enabled()

    monkeypatch.setenv("NVFP4_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_GRAD_SR_AXES", "none")
    assert not sr.localcta_v4_grad_sr_enabled()

    monkeypatch.delenv("NVFP4_SCALE_SR_GRAD", raising=False)
    monkeypatch.setenv("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", "1")
    assert sr.localcta_v4_grad_sr_enabled()


def test_non_gradient_localcta_sr_is_rejected_instead_of_using_global_atomic(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    monkeypatch.setenv("NVFP4_SR_ACTIVATION", "1")
    monkeypatch.setenv("NVFP4_SR_GRAD", "0")
    monkeypatch.setenv("NVFP4_SR_WEIGHT", "0")

    with pytest.raises(RuntimeError, match="activation/weight SR"):
        sr.build_localcta_sr_state_for_trainer(
            (),
            device="cpu",
            training_steps=10,
            gradient_accumulation_steps=1,
        )


def test_qkv_gradient_scale_sr_is_rejected_until_split3_forwards_it(
    monkeypatch,
) -> None:
    FusedAttention = type("FusedAttentionFP4_TK", (torch.nn.Module,), {})
    root = torch.nn.Module()
    root.attn = FusedAttention()
    root.attn._lbt_debug_name = "layers.0.attention"
    monkeypatch.setenv("USE_TK_LOCALCTA", "1")
    monkeypatch.setenv("USE_TK_LOCALCTA_VARIANT", "v4")
    monkeypatch.setenv("NVFP4_SR_GRAD", "0")
    monkeypatch.setenv("NVFP4_SCALE_SR_GRAD", "1")
    monkeypatch.setenv("NVFP4_SR_ACTIVATION", "0")
    monkeypatch.setenv("NVFP4_SR_WEIGHT", "0")

    with pytest.raises(RuntimeError, match="QKV split3 producer"):
        sr.build_localcta_sr_state_for_trainer(
            (root,),
            device="cpu",
            training_steps=10,
            gradient_accumulation_steps=1,
        )


def test_checkpointed_wo_sr_rejects_alternate_split2_producer() -> None:
    from low_bits_training.quantization import fused_te_linear

    with pytest.raises(RuntimeError, match="alternate WO split2"):
        fused_te_linear._validate_checkpointed_localcta_wo_sr_route(
            torch.tensor([1, 2], dtype=torch.int64),
            skip_generic_v4_wo_dy_quant=True,
        )

    fused_te_linear._validate_checkpointed_localcta_wo_sr_route(
        torch.tensor([1, 2], dtype=torch.int64),
        skip_generic_v4_wo_dy_quant=False,
    )
