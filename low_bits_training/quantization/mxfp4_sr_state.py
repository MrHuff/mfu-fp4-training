"""Checkpointed stochastic-rounding streams for MXFP4 backward producers.

The MXFP4 quantization kernels accept an explicit Philox seed and subsequence.
This module assigns those coordinates by stable converted-module identity,
rather than by process-global Python call order, and checkpoints the next
subsequence for every rank and producer.

The state is intentionally eager-only.  Passing scalar subsequences through a
captured CUDA graph would bake the capture-time values into every replay.  The
production recipes using this state therefore keep body CUDA graphs disabled.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import types
from typing import Iterable, Mapping

import torch
import torch.distributed as dist


_LOGGER = logging.getLogger(__name__)


CHECKPOINT_KEY = "mxfp4_sr_state"
STATE_VERSION = 1
SEED_NAMESPACE_VERSION = 1
SUBSEQUENCE_STRIDE = 1 << 32
UINT64_MAX = (1 << 64) - 1
DEFAULT_RESERVATION_MARGIN = 4096
MAX_RESERVATIONS_PER_PRODUCER_MICROBATCH = 3


def _require_identity(value: str | None) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            "checkpointed MXFP4 SR requires a non-empty stable _lbt_debug_name"
        )
    return value


def qkv_grad_key(debug_name: str) -> str:
    return f"{_require_identity(debug_name)}:sr:qkv_grad"


def wo_grad_key(debug_name: str) -> str:
    return f"{_require_identity(debug_name)}:sr:wo_grad"


def ffn_w2_grad_key(debug_name: str) -> str:
    return f"{_require_identity(debug_name)}:sr:ffn_w2_grad"


def ffn_deriv_grad_key(debug_name: str) -> str:
    return f"{_require_identity(debug_name)}:sr:ffn_deriv_grad"


def linear_grad_key(debug_name: str) -> str:
    return f"{_require_identity(debug_name)}:sr:linear_grad"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _grad_sr_axes() -> str:
    value = os.environ.get("MXFP4_GRAD_SR_AXES", "both")
    value = value.strip().lower().replace("-", "_")
    value = {
        "dgrad": "row",
        "rows": "row",
        "column": "col",
        "columns": "col",
        "wgrad": "col",
        "all": "both",
        "row_col": "both",
        "rowcol": "both",
        "off": "none",
        "0": "none",
    }.get(value, value)
    if value not in {"none", "row", "col", "both"}:
        raise ValueError(
            f"Unsupported MXFP4_GRAD_SR_AXES={value!r}; expected none, row, col, or both"
        )
    return value


def ranked_row_grad_sr_enabled() -> bool:
    """Return whether the proven row-gradient MXFP4 SR policy is active."""
    return _env_flag("MXFP4_SR_GRAD", False) and _grad_sr_axes() == "row"


def _as_signed_int64(value: int) -> int:
    value &= UINT64_MAX
    return value if value < (1 << 63) else value - (1 << 64)


def _as_uint64(value: int) -> int:
    return int(value) & UINT64_MAX


def _validate_uint64(name: str, value: int) -> int:
    value = int(value)
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64 - 1], got {value}")
    return value


def _checkpoint_int64_tensor(name: str, value: object) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.int64:
        raise RuntimeError(
            f"MXFP4 SR checkpoint field {name!r} must be an int64 tensor"
        )
    return value.detach().cpu()


def _checkpoint_int64_scalar(name: str, value: object) -> int:
    tensor = _checkpoint_int64_tensor(name, value)
    if tensor.numel() != 1:
        raise RuntimeError(
            f"MXFP4 SR checkpoint field {name!r} must contain one value"
        )
    return int(tensor.reshape(-1)[0].item())


def _resolve_rank_world(
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int]:
    distributed = dist.is_available() and dist.is_initialized()
    actual_rank = dist.get_rank() if distributed else 0
    actual_world_size = dist.get_world_size() if distributed else 1
    resolved_rank = actual_rank if rank is None else int(rank)
    resolved_world_size = actual_world_size if world_size is None else int(world_size)
    if resolved_world_size <= 0:
        raise ValueError(
            f"MXFP4 SR world_size must be positive, got {resolved_world_size}"
        )
    if not 0 <= resolved_rank < resolved_world_size:
        raise ValueError(
            "MXFP4 SR rank must be in [0, world_size), got "
            f"rank={resolved_rank}, world_size={resolved_world_size}"
        )
    if distributed and (
        resolved_rank != actual_rank or resolved_world_size != actual_world_size
    ):
        raise RuntimeError(
            "MXFP4 SR rank/world namespace disagrees with the active process "
            f"group: requested=({resolved_rank},{resolved_world_size}), "
            f"actual=({actual_rank},{actual_world_size})"
        )
    return resolved_rank, resolved_world_size


def discover_logical_keys(model_parts: Iterable[torch.nn.Module]) -> tuple[str, ...]:
    """Discover stable producers in the fused dense MXFP4 training path."""
    owners: dict[str, int] = {}
    attention_classes = {"FusedAttentionMXFP4_TK"}
    ffn_classes = {
        "FusedFeedForwardMXFP4_TK",
        "FusedSquaredReLUFeedForwardMXFP4_TK",
        "ExperimentalFusedSquaredReLUFeedForwardMXFP4_TK",
    }

    def add(key: str, module: torch.nn.Module) -> None:
        previous = owners.get(key)
        if previous is not None and previous != id(module):
            raise RuntimeError(
                f"duplicate MXFP4 SR logical identity {key!r}; converted modules "
                "must have unique stable _lbt_debug_name values"
            )
        owners[key] = id(module)

    for model in model_parts:
        for module in model.modules():
            class_name = module.__class__.__name__
            if class_name in attention_classes:
                base = _require_identity(getattr(module, "_lbt_debug_name", None))
                add(qkv_grad_key(f"{base}:qkv"), module)
                if not getattr(module, "_force_wo_bf16", False):
                    add(wo_grad_key(f"{base}:wo"), module)
            elif class_name in ffn_classes:
                base = _require_identity(getattr(module, "_lbt_debug_name", None))
                add(ffn_w2_grad_key(base), module)
                add(ffn_deriv_grad_key(base), module)

    return tuple(sorted(owners))


class MXFP4SRState:
    """Rank-separated, per-producer advancing Philox coordinates."""

    def __init__(
        self,
        logical_keys: Iterable[str],
        *,
        device: torch.device | str,
        user_seed: int,
        user_subsequence_base: int,
        training_steps: int,
        gradient_accumulation_steps: int,
        reservation_margin: int = DEFAULT_RESERVATION_MARGIN,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        keys = tuple(sorted(set(logical_keys)))
        if not keys:
            raise ValueError("checkpointed MXFP4 SR requires at least one logical producer")
        for key in keys:
            _require_identity(key)
        self.logical_keys = keys
        self.device = torch.device(device)
        self.rank, self.world_size = _resolve_rank_world(rank, world_size)
        self._validate_distributed_manifest()
        if len(keys) * self.world_size >= (1 << 64):
            raise ValueError("MXFP4 SR rank/producer namespace exceeds uint64")
        self.user_seed = _validate_uint64("MXFP4_SR_SEED", user_seed)
        self._active_user_seed = self.user_seed
        self.user_subsequence_base = _validate_uint64(
            "MXFP4_SR_SUBSEQUENCE", user_subsequence_base
        )
        training_steps = int(training_steps)
        gradient_accumulation_steps = int(gradient_accumulation_steps)
        reservation_margin = int(reservation_margin)
        if training_steps < 0 or gradient_accumulation_steps <= 0 or reservation_margin < 0:
            raise ValueError("invalid MXFP4 SR reservation horizon")
        self.reservations_per_slot = (
            (training_steps + 1)
            * gradient_accumulation_steps
            * MAX_RESERVATIONS_PER_PRODUCER_MICROBATCH
            + reservation_margin
        )
        span = self.reservations_per_slot * SUBSEQUENCE_STRIDE
        if self.user_subsequence_base > UINT64_MAX - span:
            raise ValueError(
                "MXFP4_SR_SUBSEQUENCE leaves insufficient uint64 headroom: "
                f"base={self.user_subsequence_base}, reservations={self.reservations_per_slot}"
            )
        self._slot_by_key = {key: slot for slot, key in enumerate(keys)}
        self._next_subsequence = {
            key: self.user_subsequence_base for key in self.logical_keys
        }
        self._lock = threading.Lock()

    def _validate_distributed_manifest(self) -> None:
        if self.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
            return
        from ..distributed_control import get_control_process_group

        process_group = get_control_process_group()
        manifests: list[object] = [None] * self.world_size
        dist.all_gather_object(
            manifests,
            self.logical_keys,
            group=process_group,
        )
        mismatched = [
            rank
            for rank, manifest in enumerate(manifests)
            if not isinstance(manifest, (tuple, list))
            or tuple(manifest) != self.logical_keys
        ]
        if mismatched:
            raise RuntimeError(
                "MXFP4 SR logical-producer manifests differ across ranks; "
                f"local_rank={self.rank}, mismatched_ranks={mismatched}"
            )
        if process_group is not None:
            digest = hashlib.sha256(
                "\n".join(self.logical_keys).encode("utf-8")
            ).hexdigest()
            _LOGGER.info(
                "MXFP4 SR manifest control receipt: backend=%s rank=%d/%d "
                "producers=%d sha256=%s",
                dist.get_backend(process_group),
                dist.get_rank(process_group),
                dist.get_world_size(process_group),
                len(self.logical_keys),
                digest,
            )

    def _seed_for(self, seed_base: int, rank: int, slot: int) -> int:
        namespace_slot = rank * len(self.logical_keys) + slot
        return (int(seed_base) + namespace_slot + 1) & UINT64_MAX

    def reserve(self, logical_key: str) -> tuple[int, int]:
        """Atomically reserve the next coordinate for one logical producer."""
        try:
            slot = self._slot_by_key[logical_key]
        except KeyError as exc:
            raise RuntimeError(
                f"unregistered MXFP4 SR logical identity {logical_key!r}; "
                "the converted model/checkpoint manifest is inconsistent"
            ) from exc
        with self._lock:
            subsequence = self._next_subsequence[logical_key]
            next_subsequence = subsequence + SUBSEQUENCE_STRIDE
            if next_subsequence > UINT64_MAX:
                raise OverflowError(
                    f"MXFP4 SR subsequence exhausted for producer {logical_key!r}"
                )
            self._next_subsequence[logical_key] = next_subsequence
        return self._seed_for(self._active_user_seed, self.rank, slot), subsequence

    def peek(self, logical_key: str) -> tuple[int, int]:
        try:
            slot = self._slot_by_key[logical_key]
        except KeyError as exc:
            raise RuntimeError(f"unregistered MXFP4 SR logical identity {logical_key!r}") from exc
        with self._lock:
            subsequence = self._next_subsequence[logical_key]
        return self._seed_for(self._active_user_seed, self.rank, slot), subsequence

    def _local_state_matrix(self) -> torch.Tensor:
        with self._lock:
            rows = [
                [
                    _as_signed_int64(
                        self._seed_for(self._active_user_seed, self.rank, slot)
                    ),
                    _as_signed_int64(self._next_subsequence[key]),
                ]
                for slot, key in enumerate(self.logical_keys)
            ]
        return torch.tensor(rows, dtype=torch.int64, device=self.device)

    def _gather_rank_states(self) -> torch.Tensor:
        local = self._local_state_matrix()
        if self.world_size == 1:
            return local.unsqueeze(0).cpu()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "multi-rank MXFP4 SR checkpoints require an initialized distributed process group"
            )
        if (dist.get_rank(), dist.get_world_size()) != (self.rank, self.world_size):
            raise RuntimeError("MXFP4 SR rank/world changed after initialization")
        from ..distributed_control import gather_checkpoint_tensor

        return gather_checkpoint_tensor(
            local,
            expected_rank=self.rank,
            expected_world_size=self.world_size,
        )

    def _checkpoint_state_dict(self, states: torch.Tensor) -> dict[str, object]:
        return {
            "version": torch.tensor([STATE_VERSION], dtype=torch.int64),
            "seed_namespace_version": torch.tensor(
                [SEED_NAMESPACE_VERSION], dtype=torch.int64
            ),
            "seed_base": torch.tensor(
                [_as_signed_int64(self._active_user_seed)], dtype=torch.int64
            ),
            "subsequence_base": torch.tensor(
                [_as_signed_int64(self.user_subsequence_base)], dtype=torch.int64
            ),
            "subsequence_stride": torch.tensor([SUBSEQUENCE_STRIDE], dtype=torch.int64),
            "world_size": torch.tensor([self.world_size], dtype=torch.int64),
            "rank_ids": torch.arange(self.world_size, dtype=torch.int64),
            "logical_keys": list(self.logical_keys),
            "states": states.detach().cpu().to(torch.int64).clone(),
        }

    def state_dict(self) -> dict[str, object]:
        return self._checkpoint_state_dict(self._gather_rank_states())

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected = {
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
        if set(state_dict) != expected:
            raise RuntimeError(
                "invalid MXFP4 SR checkpoint fields: "
                f"expected={sorted(expected)}, got={sorted(state_dict)}"
            )
        version = _checkpoint_int64_scalar("version", state_dict["version"])
        namespace = _checkpoint_int64_scalar(
            "seed_namespace_version", state_dict["seed_namespace_version"]
        )
        seed_base = _as_uint64(_checkpoint_int64_scalar("seed_base", state_dict["seed_base"]))
        subsequence_base = _as_uint64(
            _checkpoint_int64_scalar("subsequence_base", state_dict["subsequence_base"])
        )
        stride = _checkpoint_int64_scalar(
            "subsequence_stride", state_dict["subsequence_stride"]
        )
        checkpoint_world = _checkpoint_int64_scalar("world_size", state_dict["world_size"])
        if version != STATE_VERSION or namespace != SEED_NAMESPACE_VERSION:
            raise RuntimeError("unsupported MXFP4 SR checkpoint ABI")
        if stride != SUBSEQUENCE_STRIDE:
            raise RuntimeError("MXFP4 SR checkpoint stride differs from runtime")
        if checkpoint_world != self.world_size:
            raise RuntimeError(
                "MXFP4 SR checkpoint world_size differs from the runtime; refusing stream remap"
            )
        rank_ids = _checkpoint_int64_tensor("rank_ids", state_dict["rank_ids"])
        if not torch.equal(rank_ids, torch.arange(self.world_size, dtype=torch.int64)):
            raise RuntimeError("MXFP4 SR checkpoint rank namespace is malformed")
        logical_keys = tuple(state_dict["logical_keys"])
        if logical_keys != self.logical_keys:
            raise RuntimeError(
                "MXFP4 SR logical-producer manifest differs from the checkpoint; refusing stream remap"
            )
        states = _checkpoint_int64_tensor("states", state_dict["states"])
        expected_shape = (self.world_size, len(self.logical_keys), 2)
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                f"invalid MXFP4 SR state matrix shape: expected {expected_shape}, got {tuple(states.shape)}"
            )
        for rank in range(self.world_size):
            for slot in range(len(self.logical_keys)):
                expected_seed = _as_signed_int64(self._seed_for(seed_base, rank, slot))
                if int(states[rank, slot, 0].item()) != expected_seed:
                    raise RuntimeError(
                        "MXFP4 SR checkpoint seed namespace is inconsistent; refusing resume"
                    )
                next_subsequence = _as_uint64(states[rank, slot, 1].item())
                if next_subsequence < subsequence_base:
                    raise RuntimeError("MXFP4 SR checkpoint subsequence precedes its base")
                if (next_subsequence - subsequence_base) % SUBSEQUENCE_STRIDE:
                    raise RuntimeError("MXFP4 SR checkpoint subsequence is not stride-aligned")
        self._active_user_seed = seed_base
        self.user_subsequence_base = subsequence_base
        with self._lock:
            for slot, key in enumerate(self.logical_keys):
                self._next_subsequence[key] = _as_uint64(
                    states[self.rank, slot, 1].item()
                )

    def reset_to_configured_base(self) -> None:
        self._active_user_seed = self.user_seed
        with self._lock:
            self._next_subsequence = {
                key: self.user_subsequence_base for key in self.logical_keys
            }


_ACTIVE_STATE: MXFP4SRState | None = None


def active_mxfp4_sr_state() -> MXFP4SRState | None:
    return _ACTIVE_STATE


def set_active_mxfp4_sr_state(state: MXFP4SRState | None) -> None:
    global _ACTIVE_STATE
    _ACTIVE_STATE = state


def reserve_mxfp4_sr(logical_key: str | None) -> tuple[int, int]:
    if not logical_key:
        raise RuntimeError(
            "ranked row-gradient MXFP4 SR reached a quantizer without a stable producer key"
        )
    state = _ACTIVE_STATE
    if state is None:
        raise RuntimeError(
            "ranked row-gradient MXFP4 SR has no active checkpointed state; "
            "install it through the trainer before the first backward"
        )
    return state.reserve(logical_key)


def build_mxfp4_sr_state_for_trainer(
    model_parts: Iterable[torch.nn.Module],
    *,
    device: torch.device | str,
    training_steps: int,
    gradient_accumulation_steps: int,
) -> MXFP4SRState | None:
    set_active_mxfp4_sr_state(None)
    if not ranked_row_grad_sr_enabled():
        return None
    if _env_flag("MXFP4_SCALE_SR_GRAD", False):
        raise RuntimeError("ranked row-gradient MXFP4 data SR requires scale SR off")
    logical_keys = discover_logical_keys(model_parts)
    expected_raw = os.environ.get("MXFP4_SR_EXPECTED_PRODUCERS", "").strip()
    if expected_raw:
        try:
            expected = int(expected_raw)
        except ValueError as exc:
            raise RuntimeError("MXFP4_SR_EXPECTED_PRODUCERS must be an integer") from exc
        if len(logical_keys) != expected:
            raise RuntimeError(
                "MXFP4 SR producer manifest count mismatch: "
                f"expected={expected}, discovered={len(logical_keys)}"
            )
    state = MXFP4SRState(
        logical_keys,
        device=device,
        user_seed=int(os.environ.get("MXFP4_SR_SEED", "1234")),
        user_subsequence_base=int(os.environ.get("MXFP4_SR_SUBSEQUENCE", "0")),
        training_steps=training_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        reservation_margin=int(
            os.environ.get(
                "LBT_MXFP4_SR_RESERVATION_MARGIN",
                str(DEFAULT_RESERVATION_MARGIN),
            )
        ),
    )
    set_active_mxfp4_sr_state(state)
    return state


def checkpoint_mxfp4_sr_state_schema(checkpoint_id: str) -> str:
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    metadata = FileSystemReader(checkpoint_id).read_metadata()
    prefix = f"{CHECKPOINT_KEY}."
    fields = {
        key[len(prefix) :]
        for key in metadata.state_dict_metadata
        if key.startswith(prefix)
    }
    if not fields:
        return "missing"
    expected = {
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
    states_metadata = metadata.state_dict_metadata.get(f"{prefix}states")
    if fields == expected and isinstance(states_metadata, TensorStorageMetadata):
        if len(states_metadata.size) == 3:
            return "v1"
    return "unknown"


def register_with_checkpointer(checkpointer, state: MXFP4SRState, logger) -> None:
    """Register strict v1 state; no missing/legacy stream is inferred."""
    if getattr(checkpointer, "ft_manager", None) is not None:
        raise RuntimeError(
            "checkpointed MXFP4 SR is not yet compatible with TorchFT replica checkpoints"
        )
    if CHECKPOINT_KEY in checkpointer.states:
        raise RuntimeError(f"duplicate checkpoint state {CHECKPOINT_KEY!r}")
    checkpointer.states[CHECKPOINT_KEY] = state
    original_dcp_load = checkpointer.dcp_load

    def dcp_load_with_mxfp4_sr(
        this,
        state_dict,
        checkpoint_id,
        from_hf,
        from_quantized,
    ):
        if CHECKPOINT_KEY in state_dict and not from_hf:
            try:
                schema = checkpoint_mxfp4_sr_state_schema(checkpoint_id)
            except Exception as exc:
                raise RuntimeError(
                    "could not verify MXFP4 SR state in checkpoint metadata; refusing resume"
                ) from exc
            if schema != "v1":
                raise RuntimeError(
                    "MXFP4 row-gradient SR resume requires the exact checkpointed "
                    f"v1 state, got schema={schema!r}; no step/call-order inference is allowed"
                )
        result = original_dcp_load(
            state_dict,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        if CHECKPOINT_KEY in state_dict and not from_hf:
            logger.info(
                "Restored checkpointed MXFP4 SR ABI v1 for rank %d/%d before the next backward.",
                state.rank,
                state.world_size,
            )
        return result

    checkpointer.dcp_load = types.MethodType(dcp_load_with_mxfp4_sr, checkpointer)
