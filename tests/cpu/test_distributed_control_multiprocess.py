from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp


def _control_plane_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    checkpoint_path: str,
) -> None:
    os.environ["LBT_LIGHT_IMPORT"] = "1"
    os.environ["LBT_QUANTIZATION_LIGHT_IMPORT"] = "1"
    os.environ["LBT_USE_GLOO_CONTROL_PLANE"] = "1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    group = None
    try:
        from low_bits_training import distributed_control as control
        from low_bits_training.ema_checkpoint import dcp_load_helper
        from low_bits_training.quantization.mxfp4_sr_state import MXFP4SRState

        control.reset_control_process_group_for_testing()
        group = control.initialize_control_process_group()
        assert group is not None
        assert dist.get_backend(group) == "gloo"
        assert dist.get_world_size(group) == world_size

        receipt = control.validate_control_process_group()
        receipts: list[object] = [None] * world_size
        dist.all_gather_object(receipts, receipt, group=group)
        assert receipts == [receipt] * world_size

        logical_keys = ("layer.0:qkv", "layer.0:w2")
        state = MXFP4SRState(
            logical_keys,
            device="cpu",
            user_seed=1234,
            user_subsequence_base=0,
            training_steps=38_147,
            gradient_accumulation_steps=2,
            rank=rank,
            world_size=world_size,
        )
        assert len(state.logical_keys) == 2
        checkpoint_table = state._gather_rank_states()
        assert checkpoint_table.shape == (world_size, 2, 2)
        assert checkpoint_table.device.type == "cpu"
        assert checkpoint_table[:, :, 0].tolist() == [
            [1235, 1236],
            [1237, 1238],
        ]
        table_receipts: list[object] = [None] * world_size
        dist.all_gather_object(
            table_receipts,
            checkpoint_table.tolist(),
            group=group,
        )
        assert table_receipts == [checkpoint_table.tolist()] * world_size

        # Exercise the real stateful path that production DCP invokes.  Each
        # rank advances each producer by a different amount so this proves the
        # gathered full table and the rank-local row selected during restore.
        reservation_counts = (rank + 1, 3 * rank + 2)
        for key, count in zip(logical_keys, reservation_counts, strict=True):
            for _ in range(count):
                state.reserve(key)
        expected_next = {key: state.peek(key) for key in logical_keys}

        saved = {"mxfp4_sr_state": state}
        dcp.save(saved, checkpoint_id=checkpoint_path, process_group=group)
        restored = MXFP4SRState(
            logical_keys,
            device="cpu",
            user_seed=999_999,
            user_subsequence_base=17,
            training_steps=38_147,
            gradient_accumulation_steps=2,
            rank=rank,
            world_size=world_size,
        )
        loaded = {"mxfp4_sr_state": restored}
        dcp_load_helper(
            loaded,
            checkpoint_id=checkpoint_path,
            process_group=group,
        )
        for key in logical_keys:
            assert restored.peek(key) == expected_next[key]
            expected_seed, expected_subsequence = expected_next[key]
            resumed_seed, resumed_subsequence = restored.reserve(key)
            assert (resumed_seed, resumed_subsequence) == (
                expected_seed,
                expected_subsequence,
            )
            assert restored.peek(key) == (
                expected_seed,
                expected_subsequence + (1 << 32),
            )
        dist.barrier(group=group)
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def test_real_two_rank_control_manifest_and_dcp_roundtrip(tmp_path: Path) -> None:
    rendezvous = tmp_path / "gloo-rendezvous"
    checkpoint = tmp_path / "checkpoint"
    mp.spawn(
        _control_plane_worker,
        args=(2, str(rendezvous), str(checkpoint)),
        nprocs=2,
        join=True,
    )
