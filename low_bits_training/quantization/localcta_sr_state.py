"""Checkpointed stochastic-rounding state for localCTA-v4 backward producers.

The localCTA CUDA extension accepts one persistent ``int64[2]`` tensor per
logical producer.  Element zero is the Philox seed and element one is the next
subsequence.  The extension advances the subsequence on the producer's CUDA
stream, immediately before it launches the quantization kernel.

Keeping one tensor per stable module/operation identity makes assignment
independent of Python launch order and of overlap between CUDA streams.  This
module owns those tensors and makes them TorchTitan-checkpointable.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import types
from typing import Iterable, Iterator, Mapping

import torch
import torch.distributed as dist


CHECKPOINT_KEY = "localcta_sr_state"
LEGACY_STATE_VERSION = 1
STATE_VERSION = 2
SEED_NAMESPACE_VERSION = 1
SUBSEQUENCE_STRIDE = 1 << 32
UINT64_MAX = (1 << 64) - 1
DEFAULT_RESERVATION_MARGIN = 4096
V1_MIGRATION_ENV = "LBT_LOCALCTA_SR_V1_MIGRATION"
V1_RANK_NAMESPACE_MIGRATION = "rank_namespace_v2"
V1_EXPECTED_WORLD_SIZE_ENV = "LBT_LOCALCTA_SR_V1_EXPECTED_WORLD_SIZE"


def ffn_w2_grad_key(debug_name: str) -> str:
    return f"{_require_debug_name(debug_name)}:sr:ffn_w2_grad"


def ffn_deriv_grad_key(debug_name: str) -> str:
    return f"{_require_debug_name(debug_name)}:sr:ffn_deriv_grad"


def qkv_grad_key(debug_name: str) -> str:
    return f"{_require_debug_name(debug_name)}:sr:qkv_grad"


def wo_grad_key(debug_name: str) -> str:
    return f"{_require_debug_name(debug_name)}:sr:wo_grad"


def _require_debug_name(debug_name: str) -> str:
    if not isinstance(debug_name, str) or not debug_name:
        raise RuntimeError(
            "checkpointed localCTA SR requires a non-empty stable _lbt_debug_name"
        )
    return debug_name


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _localcta_v4_enabled() -> bool:
    return (
        _env_flag("USE_TK_LOCALCTA", False)
        and os.environ.get("USE_TK_LOCALCTA_VARIANT", "v1").strip().lower() == "v4"
    )


def _role_sr_enabled(role: str) -> bool:
    role = role.upper()
    explicit_data = os.environ.get(f"NVFP4_SR_{role}")
    data_sr_policy = (
        _env_flag(f"NVFP4_SR_{role}", False)
        if explicit_data is not None
        else _env_flag("NVFP4_USE_STOCHASTIC_ROUNDING", False)
    )
    if role == "GRAD":
        grad_axes = (
            os.environ.get("NVFP4_GRAD_SR_AXES", "both").strip().lower().replace("-", "_")
        )
        data_sr = data_sr_policy and grad_axes not in {"none", "off", "0"}
    else:
        data_sr = data_sr_policy

    explicit_scale = os.environ.get(f"NVFP4_SCALE_SR_{role}")
    if explicit_scale is not None:
        scale_sr = _env_flag(f"NVFP4_SCALE_SR_{role}", False)
    elif _env_flag("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", False):
        scale_sr = data_sr_policy if explicit_data is not None else True
    else:
        scale_sr = False
    return data_sr or scale_sr


def _role_scale_sr_enabled(role: str) -> bool:
    """Mirror the role-specific scale-SR enable policy exactly."""
    role = role.upper()
    explicit_scale = os.environ.get(f"NVFP4_SCALE_SR_{role}")
    if explicit_scale is not None:
        return _env_flag(f"NVFP4_SCALE_SR_{role}", False)
    if not _env_flag("NVFP4_USE_SCALE_STOCHASTIC_ROUNDING", False):
        return False
    explicit_data = os.environ.get(f"NVFP4_SR_{role}")
    if explicit_data is not None:
        return _env_flag(f"NVFP4_SR_{role}", False)
    return True


def localcta_v4_grad_sr_enabled() -> bool:
    """Mirror the localCTA-v4 gradient data/scale-SR enable policy."""
    return _localcta_v4_enabled() and _role_sr_enabled("grad")


def _as_signed_int64(value: int) -> int:
    value &= UINT64_MAX
    return value if value < (1 << 63) else value - (1 << 64)


def _as_uint64(value: int) -> int:
    return int(value) & UINT64_MAX


def _checkpoint_int64_tensor(name: str, value: object) -> torch.Tensor:
    if not torch.is_tensor(value) or value.dtype != torch.int64:
        raise RuntimeError(
            f"localCTA SR checkpoint field {name!r} must be an int64 tensor"
        )
    return value.detach().cpu()


def _checkpoint_int64_scalar(name: str, value: object) -> int:
    tensor = _checkpoint_int64_tensor(name, value)
    if tensor.numel() != 1:
        raise RuntimeError(
            f"localCTA SR checkpoint field {name!r} must contain one value"
        )
    return int(tensor.reshape(-1)[0].item())


def _validate_uint64(name: str, value: int) -> int:
    value = int(value)
    if not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be in [0, 2**64 - 1], got {value}")
    return value


def _validate_subsequence_headroom(
    base: int,
    training_steps: int,
    gradient_accumulation_steps: int,
    reservation_margin: int,
) -> int:
    training_steps = int(training_steps)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    reservation_margin = int(reservation_margin)
    if training_steps < 0:
        raise ValueError(f"training_steps must be non-negative, got {training_steps}")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, got "
            f"{gradient_accumulation_steps}"
        )
    if reservation_margin < 0:
        raise ValueError(
            f"reservation_margin must be non-negative, got {reservation_margin}"
        )

    # One reservation per logical producer and microbatch.  The extra full
    # optimizer step is conservative around end-step conventions; the margin
    # covers debug/retry calls without inferring state from a training step.
    reservations = (training_steps + 1) * gradient_accumulation_steps + reservation_margin
    span = reservations * SUBSEQUENCE_STRIDE
    if base > UINT64_MAX - span:
        raise ValueError(
            "NVFP4_RNG_SUBSEQUENCE_BASE leaves insufficient uint64 headroom: "
            f"base={base}, reservations={reservations}, stride={SUBSEQUENCE_STRIDE}, "
            f"required_span={span}"
        )
    return reservations


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
            f"localCTA SR world_size must be positive, got {resolved_world_size}"
        )
    if not 0 <= resolved_rank < resolved_world_size:
        raise ValueError(
            "localCTA SR rank must be in [0, world_size), got "
            f"rank={resolved_rank}, world_size={resolved_world_size}"
        )
    if distributed and (
        resolved_rank != actual_rank or resolved_world_size != actual_world_size
    ):
        raise RuntimeError(
            "localCTA SR rank/world namespace disagrees with the active process "
            f"group: requested=({resolved_rank},{resolved_world_size}), "
            f"actual=({actual_rank},{actual_world_size})"
        )
    return resolved_rank, resolved_world_size


def discover_logical_keys(model_parts: Iterable[torch.nn.Module]) -> tuple[str, ...]:
    """Discover stable localCTA backward producers in converted model parts."""
    owners: dict[str, int] = {}
    ffn_classes = {
        "FusedFeedForwardFP4_TK",
        "FusedSquaredReLUFeedForwardFP4_TK",
    }
    attention_classes = {"FusedAttentionFP4_TK"}

    def add(key: str, module: torch.nn.Module) -> None:
        previous = owners.get(key)
        if previous is not None and previous != id(module):
            raise RuntimeError(
                f"duplicate localCTA SR logical identity {key!r}; "
                "converted modules must have unique stable _lbt_debug_name values"
            )
        owners[key] = id(module)

    for model in model_parts:
        for module in model.modules():
            class_name = module.__class__.__name__
            if class_name not in ffn_classes | attention_classes:
                continue
            base = _require_debug_name(getattr(module, "_lbt_debug_name", None))
            if class_name in ffn_classes:
                add(ffn_w2_grad_key(base), module)
                add(ffn_deriv_grad_key(base), module)
            else:
                add(qkv_grad_key(f"{base}:qkv"), module)
                add(wo_grad_key(f"{base}:wo"), module)

    return tuple(sorted(owners))


class LocalCTASRState:
    """Per-rank, per-logical-producer counters with a strict checkpoint ABI."""

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
            raise ValueError(
                "checkpointed localCTA SR requires at least one logical producer"
            )
        for key in keys:
            _require_debug_name(key)

        self.logical_keys = keys
        self.device = torch.device(device)
        self.rank, self.world_size = _resolve_rank_world(rank, world_size)
        self._validate_distributed_manifest()
        namespace_slots = len(keys) * self.world_size
        if namespace_slots >= (1 << 64):
            raise ValueError(
                "localCTA SR rank/producer namespace exceeds uint64: "
                f"world_size={self.world_size}, logical_keys={len(keys)}"
            )
        self.user_seed = _validate_uint64("NVFP4_RNG_SEED", user_seed)
        self._active_user_seed = self.user_seed
        self.user_subsequence_base = _validate_uint64(
            "NVFP4_RNG_SUBSEQUENCE_BASE", user_subsequence_base
        )
        self.reservations_per_slot = _validate_subsequence_headroom(
            self.user_subsequence_base,
            training_steps,
            gradient_accumulation_steps,
            reservation_margin,
        )
        self._states: dict[str, torch.Tensor] = {}
        for slot, key in enumerate(keys):
            seed = self._seed_for(self._active_user_seed, self.rank, slot)
            self._states[key] = torch.tensor(
                [
                    _as_signed_int64(seed),
                    _as_signed_int64(self.user_subsequence_base),
                ],
                dtype=torch.int64,
                device=self.device,
            )

    def _validate_distributed_manifest(self) -> None:
        if self.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
            return
        from ..distributed_control import get_control_process_group

        manifests: list[object] = [None] * self.world_size
        dist.all_gather_object(
            manifests,
            self.logical_keys,
            group=get_control_process_group(),
        )
        mismatched = [
            rank
            for rank, manifest in enumerate(manifests)
            if not isinstance(manifest, (tuple, list))
            or tuple(manifest) != self.logical_keys
        ]
        if mismatched:
            raise RuntimeError(
                "localCTA SR logical-producer manifests differ across ranks; "
                f"local_rank={self.rank}, mismatched_ranks={mismatched}"
            )

    def _seed_for(self, seed_base: int, rank: int, slot: int) -> int:
        # Rank-major indexing is collision-free for the finite
        # ``world_size * logical_keys`` namespace.  Rank zero deliberately
        # retains the v1 slot seeds, which makes explicit v1 migration easy to
        # audit while decorrelating every other rank.
        namespace_slot = rank * len(self.logical_keys) + slot
        return (int(seed_base) + namespace_slot + 1) & UINT64_MAX

    def _local_state_matrix(self) -> torch.Tensor:
        return torch.stack([self._states[key] for key in self.logical_keys])

    def get(
        self, logical_key: str, device: torch.device | str | None = None
    ) -> torch.Tensor:
        try:
            state = self._states[logical_key]
        except KeyError as exc:
            raise RuntimeError(
                f"unregistered localCTA SR logical identity {logical_key!r}; "
                "the converted model/checkpoint manifest is inconsistent"
            ) from exc
        if device is not None and torch.device(device) != state.device:
            raise RuntimeError(
                f"localCTA SR state {logical_key!r} is on {state.device}, "
                f"but its producer is on {torch.device(device)}"
            )
        return state

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            # This is deliberately a checkpoint/capture-boundary cost, not a
            # per-producer event cost.  It waits for prep increments and their
            # producers on every eager side stream and for graph replays.
            torch.cuda.synchronize(self.device)

    def _gather_rank_states(self) -> torch.Tensor:
        """Return an identical all-rank snapshot on every checkpointing rank."""
        self._synchronize_device()
        local_states = self._local_state_matrix().detach().contiguous()
        if self.world_size == 1:
            return local_states.unsqueeze(0).cpu()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "multi-rank localCTA SR checkpoints require an initialized "
                "distributed process group"
            )
        actual_rank, actual_world_size = dist.get_rank(), dist.get_world_size()
        if (actual_rank, actual_world_size) != (self.rank, self.world_size):
            raise RuntimeError(
                "localCTA SR rank/world changed after initialization: "
                f"state=({self.rank},{self.world_size}), "
                f"process_group=({actual_rank},{actual_world_size})"
            )
        from ..distributed_control import gather_checkpoint_tensor

        # CPU copies are immutable snapshots.  DCP deduplicates replicated
        # tensors across ranks, so every rank must expose this identical full
        # table rather than its distinct local row.
        return gather_checkpoint_tensor(
            local_states,
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
            "subsequence_stride": torch.tensor([SUBSEQUENCE_STRIDE], dtype=torch.int64),
            "world_size": torch.tensor([self.world_size], dtype=torch.int64),
            "rank_ids": torch.arange(self.world_size, dtype=torch.int64),
            # Store exact identities, rather than hashes, so checkpoint
            # validation itself has no collision assumption.
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
            "subsequence_stride",
            "world_size",
            "rank_ids",
            "logical_keys",
            "states",
        }
        if set(state_dict) != expected:
            raise RuntimeError(
                "invalid localCTA SR checkpoint fields: "
                f"expected={sorted(expected)}, got={sorted(state_dict)}"
            )
        version_tensor = state_dict["version"]
        namespace_tensor = state_dict["seed_namespace_version"]
        seed_base_tensor = state_dict["seed_base"]
        stride_tensor = state_dict["subsequence_stride"]
        world_size_tensor = state_dict["world_size"]
        rank_ids_tensor = state_dict["rank_ids"]
        states_tensor = state_dict["states"]
        version = _checkpoint_int64_scalar("version", version_tensor)
        namespace_version = _checkpoint_int64_scalar(
            "seed_namespace_version", namespace_tensor
        )
        seed_base = _as_uint64(_checkpoint_int64_scalar("seed_base", seed_base_tensor))
        stride = _checkpoint_int64_scalar("subsequence_stride", stride_tensor)
        checkpoint_world_size = _checkpoint_int64_scalar("world_size", world_size_tensor)
        if version != STATE_VERSION:
            raise RuntimeError(
                f"unsupported localCTA SR checkpoint version {version}; "
                f"expected {STATE_VERSION}"
            )
        if namespace_version != SEED_NAMESPACE_VERSION:
            raise RuntimeError(
                "unsupported localCTA SR seed namespace version "
                f"{namespace_version}; expected {SEED_NAMESPACE_VERSION}"
            )
        if stride != SUBSEQUENCE_STRIDE:
            raise RuntimeError(
                f"localCTA SR checkpoint stride {stride} does not match "
                f"runtime stride {SUBSEQUENCE_STRIDE}"
            )
        if checkpoint_world_size != self.world_size:
            raise RuntimeError(
                "localCTA SR checkpoint world_size differs from the runtime; "
                "refusing to remap rank stochastic streams: "
                f"checkpoint={checkpoint_world_size}, runtime={self.world_size}"
            )
        rank_ids = _checkpoint_int64_tensor("rank_ids", rank_ids_tensor)
        expected_rank_ids = torch.arange(self.world_size, dtype=torch.int64)
        if not torch.equal(rank_ids, expected_rank_ids):
            raise RuntimeError(
                "localCTA SR checkpoint rank namespace is malformed; "
                "refusing to remap stochastic streams"
            )
        logical_keys = tuple(state_dict["logical_keys"])
        if logical_keys != self.logical_keys:
            raise RuntimeError(
                "localCTA SR logical-producer manifest differs from the checkpoint; "
                "refusing to remap stochastic subsequences"
            )
        states = _checkpoint_int64_tensor("states", states_tensor)
        expected_shape = (self.world_size, len(self.logical_keys), 2)
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                "invalid localCTA SR state matrix shape: "
                f"expected {expected_shape}, got {tuple(states.shape)}"
            )
        for rank in range(self.world_size):
            for slot in range(len(self.logical_keys)):
                expected_seed = _as_signed_int64(self._seed_for(seed_base, rank, slot))
                actual_seed = int(states[rank, slot, 0].item())
                if actual_seed != expected_seed:
                    raise RuntimeError(
                        "localCTA SR checkpoint seed namespace is inconsistent "
                        f"at rank={rank}, slot={slot}; refusing resume"
                    )
        self._active_user_seed = seed_base
        for slot, key in enumerate(self.logical_keys):
            self._states[key].copy_(states[self.rank, slot].to(self.device))
        self._synchronize_device()

    def migrate_from_v1_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Explicitly migrate v1's correlated seeds into the rank namespace.

        V1 exposed one same-key table per rank, which DCP treated as replicated;
        only the table retained in the checkpoint can be recovered.  Its
        counters are applied unchanged to every rank.  Rank zero keeps the old
        seed; other ranks receive deterministic disjoint seeds.  This is a
        declared RNG-namespace transition, not a training-step-based phase
        inference.
        """
        expected = {"version", "subsequence_stride", "logical_keys", "states"}
        if set(state_dict) != expected:
            raise RuntimeError(
                "invalid localCTA SR v1 checkpoint fields: "
                f"expected={sorted(expected)}, got={sorted(state_dict)}"
            )
        version_tensor = state_dict["version"]
        stride_tensor = state_dict["subsequence_stride"]
        states_tensor = state_dict["states"]
        version = _checkpoint_int64_scalar("version", version_tensor)
        stride = _checkpoint_int64_scalar("subsequence_stride", stride_tensor)
        if version != LEGACY_STATE_VERSION:
            raise RuntimeError(
                f"localCTA SR migration requires v1, got version {version}"
            )
        if stride != SUBSEQUENCE_STRIDE:
            raise RuntimeError(
                f"localCTA SR v1 stride {stride} does not match "
                f"runtime stride {SUBSEQUENCE_STRIDE}"
            )
        logical_keys = tuple(state_dict["logical_keys"])
        if logical_keys != self.logical_keys:
            raise RuntimeError(
                "localCTA SR v1 logical-producer manifest differs from the "
                "runtime; refusing migration"
            )
        states = _checkpoint_int64_tensor("states", states_tensor)
        expected_shape = (len(self.logical_keys), 2)
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                "invalid localCTA SR v1 state matrix shape: "
                f"expected {expected_shape}, got {tuple(states.shape)}"
            )
        seed_base = (_as_uint64(states[0, 0].item()) - 1) & UINT64_MAX
        for slot in range(len(self.logical_keys)):
            expected_seed = _as_signed_int64((seed_base + slot + 1) & UINT64_MAX)
            if int(states[slot, 0].item()) != expected_seed:
                raise RuntimeError(
                    "localCTA SR v1 seed slots are inconsistent; refusing migration"
                )
        self._active_user_seed = seed_base
        for slot, key in enumerate(self.logical_keys):
            migrated = torch.tensor(
                [
                    _as_signed_int64(self._seed_for(seed_base, self.rank, slot)),
                    int(states[slot, 1].item()),
                ],
                dtype=torch.int64,
                device=self.device,
            )
            self._states[key].copy_(migrated)
        self._synchronize_device()

    def reset_to_configured_base(self) -> None:
        """Start a deliberate new SR phase from the runtime seed/base."""
        self._synchronize_device()
        self._active_user_seed = self.user_seed
        for slot, key in enumerate(self.logical_keys):
            seed = self._seed_for(self.user_seed, self.rank, slot)
            initial = torch.tensor(
                [
                    _as_signed_int64(seed),
                    _as_signed_int64(self.user_subsequence_base),
                ],
                dtype=torch.int64,
                device=self.device,
            )
            self._states[key].copy_(initial)
        self._synchronize_device()

    @contextmanager
    def preserve_during_cuda_graph_capture(
        self, logical_keys: Iterable[str]
    ) -> Iterator[None]:
        """Discard synthetic warmup/capture reservations, retaining graph ABI.

        Captured prep kernels still point at the live persistent tensors and
        therefore advance them on every actual graph replay.  Only the
        synthetic calls made while constructing the graph are rolled back.
        """
        keys = tuple(dict.fromkeys(logical_keys))
        if not keys:
            yield
            return
        self._synchronize_device()
        snapshots = {key: self.get(key).detach().clone() for key in keys}
        self._synchronize_device()
        try:
            yield
        finally:
            self._synchronize_device()
            for key, snapshot in snapshots.items():
                self._states[key].copy_(snapshot)
            self._synchronize_device()


_ACTIVE_STATE: LocalCTASRState | None = None


def active_localcta_sr_state() -> LocalCTASRState | None:
    return _ACTIVE_STATE


def set_active_localcta_sr_state(state: LocalCTASRState | None) -> None:
    global _ACTIVE_STATE
    _ACTIVE_STATE = state


def get_localcta_sr_state(
    logical_key: str, device: torch.device | str | None = None
) -> torch.Tensor | None:
    state = _ACTIVE_STATE
    return None if state is None else state.get(logical_key, device)


@contextmanager
def preserve_localcta_sr_state_during_cuda_graph_capture(
    logical_keys: Iterable[str],
) -> Iterator[None]:
    state = _ACTIVE_STATE
    if state is None:
        yield
    else:
        with state.preserve_during_cuda_graph_capture(logical_keys):
            yield


def build_localcta_sr_state_for_trainer(
    model_parts: Iterable[torch.nn.Module],
    *,
    device: torch.device | str,
    training_steps: int,
    gradient_accumulation_steps: int,
) -> LocalCTASRState | None:
    set_active_localcta_sr_state(None)
    if _localcta_v4_enabled() and any(
        _role_sr_enabled(role) for role in ("activation", "weight")
    ):
        raise RuntimeError(
            "checkpointed localCTA-v4 SR currently supports gradient producers "
            "only; activation/weight SR would fall back to an uncheckpointed "
            "extension-global atomic and is therefore rejected"
        )
    if not localcta_v4_grad_sr_enabled():
        return None
    logical_keys = discover_logical_keys(model_parts)
    if _role_scale_sr_enabled("grad") and any(
        key.endswith(":sr:qkv_grad") for key in logical_keys
    ):
        # The current localCTA split3 QKV ABI forwards data SR only.  Accepting
        # scale SR here would silently ignore the requested policy (and, for a
        # scale-only policy, leave the persistent QKV counter unadvanced).
        raise RuntimeError(
            "checkpointed localCTA gradient scale SR is unsupported for the "
            "QKV split3 producer; disable NVFP4_SCALE_SR_GRAD until that ABI "
            "forwards scale stochastic rounding"
        )
    margin = int(
        os.environ.get(
            "LBT_LOCALCTA_SR_RESERVATION_MARGIN",
            str(DEFAULT_RESERVATION_MARGIN),
        )
    )
    state = LocalCTASRState(
        logical_keys,
        device=device,
        user_seed=int(os.environ.get("NVFP4_RNG_SEED", "0")),
        user_subsequence_base=int(os.environ.get("NVFP4_RNG_SUBSEQUENCE_BASE", "0")),
        training_steps=training_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        reservation_margin=margin,
    )
    set_active_localcta_sr_state(state)
    return state


def checkpoint_localcta_sr_state_schema(checkpoint_id: str) -> str:
    """Classify the localCTA SR ABI from DCP metadata only.

    The v1 and v2 state matrices have ranks two and three, respectively.  We
    also require the complete field manifest for that ABI so a partial or
    future checkpoint fails closed instead of being guessed from one key.
    """
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

    states_metadata = metadata.state_dict_metadata.get(f"{prefix}states")
    if not isinstance(states_metadata, TensorStorageMetadata):
        return "unknown"
    v1_fields = {
        "version",
        "subsequence_stride",
        "logical_keys",
        "states",
    }
    v2_fields = {
        "version",
        "seed_namespace_version",
        "seed_base",
        "subsequence_stride",
        "world_size",
        "rank_ids",
        "logical_keys",
        "states",
    }
    if fields == v1_fields and len(states_metadata.size) == 2:
        return "v1"
    if fields == v2_fields and len(states_metadata.size) == 3:
        return "v2"
    return "unknown"


def checkpoint_contains_localcta_sr_state(checkpoint_id: str) -> bool:
    """Compatibility helper for callers that only need presence."""
    return checkpoint_localcta_sr_state_schema(checkpoint_id) != "missing"


class _LegacyV1LoadState:
    """DCP destination for an explicitly authorized v1-to-v2 migration."""

    def __init__(self, logical_keys: tuple[str, ...]) -> None:
        self.logical_keys = logical_keys
        self.loaded_state_dict: dict[str, object] | None = None

    def state_dict(self) -> dict[str, object]:
        return {
            "version": torch.zeros(1, dtype=torch.int64),
            "subsequence_stride": torch.zeros(1, dtype=torch.int64),
            "logical_keys": list(self.logical_keys),
            "states": torch.empty((len(self.logical_keys), 2), dtype=torch.int64),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        snapshot: dict[str, object] = {}
        for key, value in state_dict.items():
            snapshot[key] = (
                value.detach().cpu().clone() if torch.is_tensor(value) else value
            )
        self.loaded_state_dict = snapshot


def register_with_checkpointer(checkpointer, state: LocalCTASRState, logger) -> None:
    """Register state and make pre-state checkpoints explicitly compatible.

    DCP rejects a requested state key that is absent from an older checkpoint.
    We omit only this new key after verifying metadata.  A v1 checkpoint is
    rejected unless the operator explicitly requests the documented v2 rank
    namespace migration.  No path infers stochastic state from a training
    step.
    """
    if getattr(checkpointer, "ft_manager", None) is not None:
        # TorchTitan installs its TorchFT state closures in CheckpointManager's
        # constructor and hard-codes MODEL/OPTIMIZER/LR/TRAIN_STATE there.  A
        # replica failover would silently omit this later-added custom state.
        raise RuntimeError(
            "checkpointed localCTA SR is not yet compatible with TorchFT "
            "replica checkpoints; disable TorchFT or extend its state closure "
            "before enabling localCTA-v4 gradient SR"
        )
    if CHECKPOINT_KEY in checkpointer.states:
        raise RuntimeError(f"duplicate checkpoint state {CHECKPOINT_KEY!r}")
    checkpointer.states[CHECKPOINT_KEY] = state
    original_dcp_load = checkpointer.dcp_load

    def dcp_load_with_localcta_sr_compat(
        this,
        state_dict,
        checkpoint_id,
        from_hf,
        from_quantized,
    ):
        load_state = state_dict
        starts_new_sr_phase = False
        legacy_v1_loader = None
        if CHECKPOINT_KEY in state_dict and not from_hf:
            try:
                schema = checkpoint_localcta_sr_state_schema(checkpoint_id)
            except Exception as exc:
                raise RuntimeError(
                    "could not verify localCTA SR state in checkpoint metadata; "
                    "refusing an ambiguous resume"
                ) from exc
            if schema == "missing":
                load_state = dict(state_dict)
                load_state.pop(CHECKPOINT_KEY)
                starts_new_sr_phase = True
            elif schema == "v1":
                migration = os.environ.get(V1_MIGRATION_ENV, "").strip().lower()
                if migration != V1_RANK_NAMESPACE_MIGRATION:
                    detail = (
                        f"unsupported value {migration!r} for {V1_MIGRATION_ENV}; "
                        if migration
                        else ""
                    )
                    raise RuntimeError(
                        "localCTA SR checkpoint uses the rank-correlated v1 RNG "
                        "namespace. Resume is rejected by default. "
                        f"{detail}Set {V1_MIGRATION_ENV}="
                        f"{V1_RANK_NAMESPACE_MIGRATION} to explicitly preserve "
                        "persisted subsequence counters while moving each rank "
                        "into the v2 seed namespace. This transition is not "
                        "bitwise-continuous on ranks greater than zero and does "
                        "not infer state from the checkpoint step."
                    )
                expected_world_size_raw = os.environ.get(
                    V1_EXPECTED_WORLD_SIZE_ENV, ""
                ).strip()
                if not expected_world_size_raw:
                    raise RuntimeError(
                        "localCTA SR v1 did not checkpoint world_size. Set "
                        f"{V1_EXPECTED_WORLD_SIZE_ENV} to the verified legacy "
                        "world size before migration; refusing to infer it."
                    )
                try:
                    expected_world_size = int(expected_world_size_raw)
                except ValueError as exc:
                    raise RuntimeError(
                        f"invalid {V1_EXPECTED_WORLD_SIZE_ENV}="
                        f"{expected_world_size_raw!r}; expected a positive integer"
                    ) from exc
                if expected_world_size <= 0:
                    raise RuntimeError(
                        f"invalid {V1_EXPECTED_WORLD_SIZE_ENV}="
                        f"{expected_world_size_raw!r}; expected a positive integer"
                    )
                if expected_world_size != state.world_size:
                    raise RuntimeError(
                        "operator-verified localCTA SR v1 world size differs "
                        "from the live runtime; refusing to remap rank streams: "
                        f"verified_v1={expected_world_size}, "
                        f"runtime={state.world_size}"
                    )
                legacy_v1_loader = _LegacyV1LoadState(state.logical_keys)
                load_state = dict(state_dict)
                load_state[CHECKPOINT_KEY] = legacy_v1_loader
            elif schema != "v2":
                raise RuntimeError(
                    "unrecognized localCTA SR checkpoint metadata; refusing an "
                    "ambiguous or partial state load"
                )
        result = original_dcp_load(
            load_state,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        if starts_new_sr_phase:
            # Do this after a successful DCP load so a failed/partial load does
            # not pretend that the requested legacy resume completed.  An
            # explicit reset matters for repeated loads in one process: merely
            # omitting the missing key would otherwise retain advanced state.
            state.reset_to_configured_base()
            logger.warning(
                "Checkpoint predates checkpointed localCTA SR state; starting "
                "an explicit new SR phase at NVFP4_RNG_SUBSEQUENCE_BASE. "
                "This resume is not bitwise-continuous and no step-based "
                "subsequence inference was performed."
            )
        if legacy_v1_loader is not None:
            if legacy_v1_loader.loaded_state_dict is None:
                raise RuntimeError(
                    "localCTA SR v1 migration loader was not populated by DCP; "
                    "refusing resume"
                )
            state.migrate_from_v1_state_dict(legacy_v1_loader.loaded_state_dict)
            logger.warning(
                "Explicitly migrated localCTA SR v1 to the rank-namespaced v2 "
                "ABI for rank %d/%d. Persisted next-subsequence counters were "
                "preserved exactly from the single v1 table retained by DCP "
                "and applied to every rank; rank zero retains its v1 seed and "
                "ranks greater than zero enter deterministic disjoint seed streams. "
                "The operator-verified legacy world size was %d. "
                "This is not bitwise-continuous on ranks greater than zero and "
                "no step-based phase inference was performed.",
                state.rank,
                state.world_size,
                state.world_size,
            )
        return result

    checkpointer.dcp_load = types.MethodType(
        dcp_load_with_localcta_sr_compat, checkpointer
    )
