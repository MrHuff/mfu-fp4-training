from __future__ import annotations

import os

import pytest
import torch


def _load(monkeypatch):
    monkeypatch.setenv("LBT_LIGHT_IMPORT", "1")
    monkeypatch.setenv("LBT_QUANTIZATION_LIGHT_IMPORT", "1")
    from low_bits_training.quantization import mxfp4_fused_linear as mxfp4
    from low_bits_training.quantization import mxfp4_sr_state as sr

    sr.set_active_mxfp4_sr_state(None)
    return mxfp4, sr


def _ranked_row_grad_sr(monkeypatch) -> None:
    monkeypatch.setenv("MXFP4_USE_STOCHASTIC_ROUNDING", "1")
    monkeypatch.setenv("MXFP4_SR_ACTIVATION", "0")
    monkeypatch.setenv("MXFP4_SR_GRAD", "1")
    monkeypatch.setenv("MXFP4_GRAD_SR_AXES", "row")
    monkeypatch.setenv("MXFP4_SR_WEIGHT", "0")
    monkeypatch.setenv("MXFP4_USE_SCALE_STOCHASTIC_ROUNDING", "0")
    monkeypatch.setenv("MXFP4_SCALE_SR_ACTIVATION", "0")
    monkeypatch.setenv("MXFP4_SCALE_SR_GRAD", "0")
    monkeypatch.setenv("MXFP4_SCALE_SR_WEIGHT", "0")
    monkeypatch.setenv("MXFP4_SR_SEED", "1234")
    monkeypatch.setenv("MXFP4_SR_SUBSEQUENCE", "17")


def _paired_col_rht(monkeypatch) -> None:
    monkeypatch.setenv("MXFP4_USE_RHT", "1")
    monkeypatch.setenv("MXFP4_RHT_TE_STYLE", "1")
    monkeypatch.setenv("MXFP4_RHT_ACTIVATION", "1")
    monkeypatch.setenv("MXFP4_RHT_GRAD", "1")
    monkeypatch.setenv("MXFP4_RHT_WEIGHT", "0")
    monkeypatch.setenv("MXFP4_RHT_AXES", "col")
    monkeypatch.setenv("MXFP4_RHT_BLOCK_SIZE", "16")
    monkeypatch.setenv("MXFP4_RHT_RANDOM_SIGN_MASK", "1")


def _state(sr, keys=("layers.0:a", "layers.0:b"), **overrides):
    kwargs = {
        "device": "cpu",
        "user_seed": 1234,
        "user_subsequence_base": 17,
        "training_steps": 71_526,
        "gradient_accumulation_steps": 4,
        "reservation_margin": 4096,
    }
    kwargs.update(overrides)
    return sr.MXFP4SRState(keys, **kwargs)


def _all_rank_checkpoint(*states):
    table = torch.stack([state._local_state_matrix().cpu() for state in states])
    return states[0]._checkpoint_state_dict(table)


def test_rank_and_producer_namespaces_do_not_alias(monkeypatch) -> None:
    _, sr = _load(monkeypatch)
    states = [
        _state(sr, keys=("z", "a"), rank=rank, world_size=4)
        for rank in range(4)
    ]
    coordinates = [
        state.peek(key)
        for state in states
        for key in state.logical_keys
    ]

    assert len(coordinates) == len(set(coordinates))
    assert states[0].peek("a") == (1235, 17)
    assert states[0].peek("z") == (1236, 17)
    assert states[3].peek("a") == (1241, 17)
    assert states[3].peek("z") == (1242, 17)


def test_per_producer_streams_ignore_cross_producer_call_order_and_repeat_calls(
    monkeypatch,
) -> None:
    _, sr = _load(monkeypatch)
    left = _state(sr)
    right = _state(sr)

    left_a = [left.reserve("layers.0:a"), left.reserve("layers.0:a")]
    left_b = [left.reserve("layers.0:b"), left.reserve("layers.0:b")]
    right_b0 = right.reserve("layers.0:b")
    right_a0 = right.reserve("layers.0:a")
    right_b1 = right.reserve("layers.0:b")
    right_a1 = right.reserve("layers.0:a")

    assert left_a == [right_a0, right_a1]
    assert left_b == [right_b0, right_b1]
    assert left_a[1][1] - left_a[0][1] == sr.SUBSEQUENCE_STRIDE
    # This covers repeated invocations such as activation-checkpoint
    # recomputation: only that stable producer advances; no other slot shifts.
    assert left.peek("layers.0:a") == right.peek("layers.0:a")
    assert left.peek("layers.0:b") == right.peek("layers.0:b")


def test_resume_matches_uninterrupted_at_next_quantized_backward(monkeypatch) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    key_a, key_b = "layers.0:a", "layers.0:b"
    uninterrupted = _state(sr, keys=(key_a, key_b), rank=1, world_size=2)
    sr.set_active_mxfp4_sr_state(uninterrupted)
    for _ in range(4):
        mxfp4._mxfp4_opt_kwargs("grad", key_a)
        mxfp4._mxfp4_opt_kwargs("grad", key_b)
    checkpoint = _all_rank_checkpoint(
        _state(sr, keys=(key_a, key_b), rank=0, world_size=2),
        uninterrupted,
    )

    expected_a = mxfp4._mxfp4_opt_kwargs("grad", key_a)
    expected_b = mxfp4._mxfp4_opt_kwargs("grad", key_b)

    restored = _state(
        sr,
        keys=(key_a, key_b),
        rank=1,
        world_size=2,
        user_seed=999,
        user_subsequence_base=999,
    )
    restored.load_state_dict(checkpoint)
    sr.set_active_mxfp4_sr_state(restored)

    assert mxfp4._mxfp4_opt_kwargs("grad", key_a) == expected_a
    assert mxfp4._mxfp4_opt_kwargs("grad", key_b) == expected_b


def test_torch_distributed_checkpoint_roundtrip_and_schema(monkeypatch, tmp_path) -> None:
    _, sr = _load(monkeypatch)
    import torch.distributed.checkpoint as dcp

    original = _state(sr)
    original.reserve("layers.0:a")
    original.reserve("layers.0:b")
    original.reserve("layers.0:b")
    checkpoint_id = str(tmp_path / "step-1000")
    dcp.save({sr.CHECKPOINT_KEY: original}, checkpoint_id=checkpoint_id)

    restored = _state(sr, user_seed=999, user_subsequence_base=999)
    dcp.load({sr.CHECKPOINT_KEY: restored}, checkpoint_id=checkpoint_id)

    assert restored.peek("layers.0:a") == original.peek("layers.0:a")
    assert restored.peek("layers.0:b") == original.peek("layers.0:b")
    assert sr.checkpoint_mxfp4_sr_state_schema(checkpoint_id) == "v1"
    assert set(original.state_dict()) == {
        "version",
        "seed_namespace_version",
        "seed_base",
        "subsequence_base",
        "subsequence_stride",
        "world_size",
        "rank_ids",
        "logical_keys",
        "states",
    }


def test_resume_rejects_world_manifest_seed_and_counter_corruption(monkeypatch) -> None:
    _, sr = _load(monkeypatch)
    rank0 = _state(sr, rank=0, world_size=2)
    rank1 = _state(sr, rank=1, world_size=2)
    checkpoint = _all_rank_checkpoint(rank0, rank1)

    with pytest.raises(RuntimeError, match="world_size differs"):
        _state(sr).load_state_dict(checkpoint)

    bad_manifest = dict(checkpoint)
    bad_manifest["logical_keys"] = ["different", "keys"]
    with pytest.raises(RuntimeError, match="manifest differs"):
        rank0.load_state_dict(bad_manifest)

    bad_seed = dict(checkpoint)
    bad_seed["states"] = checkpoint["states"].clone()
    bad_seed["states"][1, 0, 0] += 1
    with pytest.raises(RuntimeError, match="seed namespace"):
        rank0.load_state_dict(bad_seed)

    bad_counter = dict(checkpoint)
    bad_counter["states"] = checkpoint["states"].clone()
    bad_counter["states"][0, 0, 1] += 1
    with pytest.raises(RuntimeError, match="stride-aligned"):
        rank0.load_state_dict(bad_counter)


def test_row_grad_sr_fails_closed_without_state_or_stable_key(monkeypatch) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    key = "layers.0:a"

    with pytest.raises(RuntimeError, match="stable producer key"):
        mxfp4._mxfp4_opt_kwargs("grad")

    with pytest.raises(RuntimeError, match="no active checkpointed state"):
        mxfp4._mxfp4_opt_kwargs("grad", key)

    sr.set_active_mxfp4_sr_state(_state(sr, keys=(key,)))
    assert mxfp4._mxfp4_opt_kwargs("grad", key)["rng_subsequence"] == 17


def test_sr_off_has_no_route_or_rng_drift(monkeypatch) -> None:
    mxfp4, _ = _load(monkeypatch)
    monkeypatch.setenv("MXFP4_SR_ACTIVATION", "0")
    monkeypatch.setenv("MXFP4_SR_GRAD", "0")
    monkeypatch.setenv("MXFP4_SR_WEIGHT", "0")
    monkeypatch.setenv("MXFP4_SCALE_SR_ACTIVATION", "0")
    monkeypatch.setenv("MXFP4_SCALE_SR_GRAD", "0")
    monkeypatch.setenv("MXFP4_SCALE_SR_WEIGHT", "0")
    calls = []
    outputs = tuple(torch.empty(1) for _ in range(4))

    def deterministic(tensor, mode):
        calls.append((tensor.shape, mode))
        return outputs

    monkeypatch.setattr(mxfp4, "mxfp4_quantize_row_and_col", deterministic)
    result = mxfp4._quantize_row_col_bf16(
        torch.empty((128, 128), dtype=torch.bfloat16), role="grad"
    )

    assert calls == [(torch.Size([128, 128]), 1)]
    assert (
        result.row_fp4,
        result.row_sc,
        result.col_fp4,
        result.col_sc,
    ) == outputs
    assert mxfp4._mxfp4_opt_kwargs("grad") == {
        "data_stochastic_rounding": False,
        "scale_stochastic_rounding": False,
        "rng_seed": 1234,
        "rng_subsequence": 0,
    }


def test_row_only_grad_sr_keeps_shared_2d_col_orientation_deterministic(
    monkeypatch,
) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    key = "layers.0:a"
    sr.set_active_mxfp4_sr_state(_state(sr, keys=(key,)))
    calls = []
    row = (torch.empty(1), torch.empty(1))
    col = (torch.empty(1), torch.empty(1))

    def row_opt(tensor, mode, **kwargs):
        calls.append(("row_sr", kwargs))
        return row

    def col_rte(tensor, mode):
        calls.append(("col_rte", {}))
        return col

    monkeypatch.setattr(mxfp4, "mxfp4_quantize_for_gemm_opt", row_opt)
    monkeypatch.setattr(mxfp4, "mxfp4_quantize_col_only", col_rte)
    result = mxfp4._quantize_row_col_bf16(
        torch.empty((128, 128), dtype=torch.bfloat16),
        role="grad",
        producer_key=key,
    )

    assert result.row_fp4 is row[0] and result.row_sc is row[1]
    assert result.col_fp4 is col[0] and result.col_sc is col[1]
    assert calls[0][0] == "row_sr"
    assert calls[0][1]["data_stochastic_rounding"] is True
    assert calls[0][1]["scale_stochastic_rounding"] is False
    assert calls[1] == ("col_rte", {})


def test_row_grad_sr_and_paired_col_rht_use_disjoint_orientation_contracts(
    monkeypatch,
) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    _paired_col_rht(monkeypatch)
    key = "layers.0:a"
    state = _state(sr, keys=(key,))
    sr.set_active_mxfp4_sr_state(state)
    before = state.peek(key)
    calls = []
    row = (torch.empty(1), torch.empty(1))
    col = (torch.empty(1), torch.empty(1))

    def row_opt(tensor, mode, **kwargs):
        calls.append(("row_sr", kwargs))
        return row

    def col_rht(tensor, mode, **kwargs):
        calls.append(("col_rht", kwargs))
        return col

    monkeypatch.setattr(mxfp4, "mxfp4_quantize_for_gemm_opt", row_opt)
    monkeypatch.setattr(mxfp4, "mxfp4_quantize_col_only_opt_rht", col_rht)
    result = mxfp4._quantize_row_col_bf16(
        torch.empty((128, 128), dtype=torch.bfloat16),
        role="grad",
        producer_key=key,
    )

    assert result.row_fp4 is row[0] and result.row_sc is row[1]
    assert result.col_fp4 is col[0] and result.col_sc is col[1]
    assert calls[0][0] == "row_sr"
    assert calls[0][1]["data_stochastic_rounding"] is True
    assert calls[0][1]["scale_stochastic_rounding"] is False
    assert calls[1][0] == "col_rht"
    assert calls[1][1] == {
        "data_stochastic_rounding": False,
        "scale_stochastic_rounding": False,
        "rng_seed": 1234,
        "rng_subsequence": 17,
        "rht_axes": "col",
        "rht_block_size": 16,
        "with_random_sign_mask": True,
    }
    after = state.peek(key)
    assert after[0] == before[0]
    assert after[1] - before[1] == sr.SUBSEQUENCE_STRIDE


def test_paired_col_rht_helper_does_not_consume_row_sr_state(monkeypatch) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    _paired_col_rht(monkeypatch)
    key = "layers.0:a"
    state = _state(sr, keys=(key,))
    sr.set_active_mxfp4_sr_state(state)
    before = state.peek(key)
    col = (torch.empty(1), torch.empty(1))
    calls = []

    def col_rht(tensor, mode, **kwargs):
        calls.append(kwargs)
        return col

    monkeypatch.setattr(mxfp4, "mxfp4_quantize_col_only_opt_rht", col_rht)
    result = mxfp4._quantize_col_bf16(
        torch.empty((128, 128), dtype=torch.bfloat16),
        role="grad",
        producer_key=key,
    )

    assert result == col
    assert calls[0]["data_stochastic_rounding"] is False
    assert calls[0]["scale_stochastic_rounding"] is False
    assert calls[0]["rht_axes"] == "col"
    assert state.peek(key) == before


def test_row_only_grad_sr_keeps_split2_col_orientation_deterministic(
    monkeypatch,
) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    key = "layers.0:ffn"
    sr.set_active_mxfp4_sr_state(_state(sr, keys=(key,)))
    calls = []

    monkeypatch.setattr(
        mxfp4,
        "mxfp4_quantize_split2_row_only_opt_launch_inplace",
        lambda *args, **kwargs: calls.append(("row_sr", kwargs)),
    )
    monkeypatch.setattr(
        mxfp4,
        "mxfp4_quantize_split2_col_only_launch_inplace",
        lambda *args, **kwargs: calls.append(("col_rte", kwargs)),
    )
    tensor = torch.empty((128, 128), dtype=torch.bfloat16)
    payload = torch.empty((128, 128), dtype=torch.uint8)
    scale = torch.empty((1, 1, 32, 16), dtype=torch.uint8)

    mxfp4._quantize_split2_row_and_col_inplace(
        tensor,
        tensor,
        payload,
        scale,
        payload,
        scale,
        role="grad",
        producer_key=key,
    )

    assert calls[0][0] == "row_sr"
    assert calls[0][1]["rng_seed"] == 1235
    assert calls[0][1]["rng_subsequence"] == 17
    assert calls[1] == ("col_rte", {"mode": 1})


def test_row_grad_sr_and_paired_col_rht_split2_contract(monkeypatch) -> None:
    mxfp4, sr = _load(monkeypatch)
    _ranked_row_grad_sr(monkeypatch)
    _paired_col_rht(monkeypatch)
    key = "layers.0:ffn"
    state = _state(sr, keys=(key,))
    sr.set_active_mxfp4_sr_state(state)
    before = state.peek(key)
    calls = []

    monkeypatch.setattr(
        mxfp4,
        "mxfp4_quantize_split2_row_only_opt_launch_inplace",
        lambda *args, **kwargs: calls.append(("row_sr", kwargs)),
    )
    monkeypatch.setattr(
        mxfp4,
        "mxfp4_quantize_split2_col_only_opt_launch_inplace",
        lambda *args, **kwargs: calls.append(("col_rht", kwargs)),
    )
    tensor = torch.empty((128, 128), dtype=torch.bfloat16)
    payload = torch.empty((128, 128), dtype=torch.uint8)
    scale = torch.empty((1, 1, 32, 16), dtype=torch.uint8)

    mxfp4._quantize_split2_row_and_col_inplace(
        tensor,
        tensor,
        payload,
        scale,
        payload,
        scale,
        role="grad",
        producer_key=key,
    )

    assert calls[0][0] == "row_sr"
    assert calls[0][1]["data_stochastic_rounding"] is True
    assert calls[1][0] == "col_rht"
    assert calls[1][1]["data_stochastic_rounding"] is False
    assert calls[1][1]["scale_stochastic_rounding"] is False
    assert calls[1][1]["use_rht"] is True
    assert calls[1][1]["rht_block_size"] == 16
    assert calls[1][1]["with_random_sign_mask"] is True
    after = state.peek(key)
    assert after[0] == before[0]
    assert after[1] - before[1] == sr.SUBSEQUENCE_STRIDE


def test_stable_manifest_counts_for_pure32_and_tail5(monkeypatch) -> None:
    _, sr = _load(monkeypatch)
    FusedAttention = type("FusedAttentionMXFP4_TK", (torch.nn.Module,), {})
    FusedFFN = type("FusedFeedForwardMXFP4_TK", (torch.nn.Module,), {})

    def model(layers: int):
        root = torch.nn.Module()
        for idx in range(layers):
            layer = torch.nn.Module()
            layer.attention = FusedAttention()
            layer.attention._lbt_debug_name = f"layers.{idx}.attention"
            layer.ffn = FusedFFN()
            layer.ffn._lbt_debug_name = f"layers.{idx}.feed_forward"
            root.add_module(f"layer_{idx}", layer)
        return root

    pure = sr.discover_logical_keys((model(32),))
    tail5 = sr.discover_logical_keys((model(5),))

    assert len(pure) == 128
    assert len(tail5) == 20
    assert pure == tuple(sorted(pure))
    assert len(set(pure)) == len(pure)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


class _Checkpointer:
    def __init__(self):
        self.states = {"other": object()}
        self.ft_manager = None
        self.loaded = None

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        self.loaded = (state_dict, checkpoint_id, from_hf, from_quantized)
        return "loaded"


def test_checkpointer_requires_exact_state_and_logs_restore_before_backward(
    monkeypatch,
) -> None:
    _, sr = _load(monkeypatch)
    state = _state(sr)
    checkpointer = _Checkpointer()
    logger = _Logger()
    sr.register_with_checkpointer(checkpointer, state, logger)

    monkeypatch.setattr(sr, "checkpoint_mxfp4_sr_state_schema", lambda _: "missing")
    with pytest.raises(RuntimeError, match="exact checkpointed v1 state"):
        checkpointer.dcp_load(
            {sr.CHECKPOINT_KEY: state}, "/old", False, False
        )
    assert checkpointer.loaded is None

    monkeypatch.setattr(sr, "checkpoint_mxfp4_sr_state_schema", lambda _: "v1")
    assert checkpointer.dcp_load(
        {sr.CHECKPOINT_KEY: state}, "/new", False, False
    ) == "loaded"
    assert "before the next backward" in logger.messages[-1]


def test_body_cuda_graph_capture_is_not_part_of_checkpointed_scalar_abi(
    monkeypatch,
) -> None:
    _, sr = _load(monkeypatch)
    assert "captured CUDA graph" in (sr.__doc__ or "")
    # The launch capsule separately asserts graph capture is disabled.  This
    # unit guard documents why: reserve() returns changing Python scalars.
    state = _state(sr, keys=("a",))
    assert state.reserve("a") != state.reserve("a")
