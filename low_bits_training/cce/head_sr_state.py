"""Versioned, checkpointed stochastic state for the fused output-head G row.

The supported production route emits an MXFP8 G row from the fused softmax
producer.  Its CUDA ABI accepts one persistent ``int64[2]`` tensor:

``[rank/logical-producer seed, next Philox subsequence]``.

The CUDA prep kernel reserves one subsequence stride on the producer stream for
every microbatch.  This module owns that tensor, validates its optimizer-step /
gradient-accumulation progress, and makes a full all-rank snapshot visible to
TorchTitan distributed checkpointing.
"""

from __future__ import annotations

import os
import types
from typing import Callable, Mapping

import torch
import torch.distributed as dist


CHECKPOINT_KEY = "output_head_sr_state"
STATE_VERSION = 1
SEED_NAMESPACE_VERSION = 1
SUBSEQUENCE_STRIDE = 1 << 40
UINT64_MAX = (1 << 64) - 1
LOGICAL_KEYS = ("output_head:g_fused_mxfp8_row",)
DEFAULT_RESERVATION_MARGIN = 1024
MISSING_POLICY_ENV = "LBT_OUTPUT_HEAD_SR_MISSING_POLICY"
START_NEW_PHASE_POLICY = "start_new_phase"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


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
            f"output-head SR checkpoint field {name!r} must be an int64 tensor"
        )
    return value.detach().cpu()


def _checkpoint_int64_scalar(name: str, value: object) -> int:
    tensor = _checkpoint_int64_tensor(name, value)
    if tensor.numel() != 1:
        raise RuntimeError(
            f"output-head SR checkpoint field {name!r} must contain one value"
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
            f"output-head SR world_size must be positive, got {resolved_world_size}"
        )
    if not 0 <= resolved_rank < resolved_world_size:
        raise ValueError(
            "output-head SR rank must be in [0, world_size), got "
            f"rank={resolved_rank}, world_size={resolved_world_size}"
        )
    if distributed and (
        resolved_rank != actual_rank or resolved_world_size != actual_world_size
    ):
        raise RuntimeError(
            "output-head SR rank/world namespace disagrees with the active "
            f"process group: requested=({resolved_rank},{resolved_world_size}), "
            f"actual=({actual_rank},{actual_world_size})"
        )
    return resolved_rank, resolved_world_size


class OutputHeadSRState:
    """One collision-free per-rank stream with a strict checkpoint ABI."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        user_seed: int,
        user_subsequence_base: int,
        training_steps: int,
        gradient_accumulation_steps: int,
        step_getter: Callable[[], int],
        reservation_margin: int = DEFAULT_RESERVATION_MARGIN,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        self.logical_keys = LOGICAL_KEYS
        self.device = torch.device(device)
        self.rank, self.world_size = _resolve_rank_world(rank, world_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        self._step_getter = step_getter
        self.user_seed = _validate_uint64("output-head SR seed", user_seed)
        self._active_user_seed = self.user_seed
        self.user_subsequence_base = _validate_uint64(
            "output-head SR subsequence base", user_subsequence_base
        )
        self._active_subsequence_base = self.user_subsequence_base
        self.phase_origin_step = 0

        training_steps = int(training_steps)
        reservation_margin = int(reservation_margin)
        if training_steps < 0:
            raise ValueError("training_steps must be non-negative")
        if reservation_margin < 0:
            raise ValueError("reservation_margin must be non-negative")
        self.reservations_per_rank = (
            training_steps + 1
        ) * self.gradient_accumulation_steps + reservation_margin
        required_span = self.reservations_per_rank * SUBSEQUENCE_STRIDE
        if self.user_subsequence_base > UINT64_MAX - required_span:
            raise ValueError(
                "output-head SR subsequence base leaves insufficient uint64 "
                f"headroom: base={self.user_subsequence_base}, "
                f"reservations={self.reservations_per_rank}, "
                f"stride={SUBSEQUENCE_STRIDE}"
            )

        self._validate_distributed_manifest()
        self._states = {
            LOGICAL_KEYS[0]: torch.tensor(
                [
                    _as_signed_int64(self._seed_for(self.user_seed, self.rank, 0)),
                    _as_signed_int64(self.user_subsequence_base),
                ],
                dtype=torch.int64,
                device=self.device,
            )
        }

    def _validate_distributed_manifest(self) -> None:
        if self.world_size == 1 or not (dist.is_available() and dist.is_initialized()):
            return
        from ..distributed_control import get_control_process_group

        local_manifest = (
            self.logical_keys,
            self.gradient_accumulation_steps,
            SUBSEQUENCE_STRIDE,
        )
        manifests: list[object] = [None] * self.world_size
        dist.all_gather_object(
            manifests,
            local_manifest,
            group=get_control_process_group(),
        )
        mismatched = [
            rank
            for rank, manifest in enumerate(manifests)
            if manifest != local_manifest
        ]
        if mismatched:
            raise RuntimeError(
                "output-head SR manifests differ across ranks; "
                f"local_rank={self.rank}, mismatched_ranks={mismatched}"
            )

    def _seed_for(self, seed_base: int, rank: int, slot: int) -> int:
        namespace_slot = rank * len(self.logical_keys) + slot
        return (int(seed_base) + namespace_slot + 1) & UINT64_MAX

    def get(
        self,
        logical_key: str = LOGICAL_KEYS[0],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        try:
            state = self._states[logical_key]
        except KeyError as exc:
            raise RuntimeError(
                f"unregistered output-head SR logical identity {logical_key!r}"
            ) from exc
        if device is not None and torch.device(device) != state.device:
            raise RuntimeError(
                f"output-head SR state is on {state.device}, producer is on "
                f"{torch.device(device)}"
            )
        return state

    def _synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _local_state_matrix(self) -> torch.Tensor:
        return torch.stack([self._states[key] for key in self.logical_keys])

    def _gather_rank_states(self) -> torch.Tensor:
        self._synchronize_device()
        local_states = self._local_state_matrix().detach().contiguous()
        if self.world_size == 1:
            return local_states.unsqueeze(0).cpu()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "multi-rank output-head SR checkpoints require an initialized "
                "distributed process group"
            )
        if (dist.get_rank(), dist.get_world_size()) != (self.rank, self.world_size):
            raise RuntimeError("output-head SR rank/world changed after initialization")
        from ..distributed_control import gather_checkpoint_tensor

        return gather_checkpoint_tensor(
            local_states,
            expected_rank=self.rank,
            expected_world_size=self.world_size,
        )

    def _expected_next_subsequence(self, step: int) -> int:
        step = int(step)
        if step < self.phase_origin_step:
            raise RuntimeError(
                "output-head SR training step precedes its phase origin: "
                f"step={step}, origin={self.phase_origin_step}"
            )
        invocations = (step - self.phase_origin_step) * self.gradient_accumulation_steps
        expected = self._active_subsequence_base + invocations * SUBSEQUENCE_STRIDE
        if expected > UINT64_MAX:
            raise RuntimeError("output-head SR subsequence overflow")
        return expected

    def validate_progress(
        self,
        step: int | None = None,
        states: torch.Tensor | None = None,
    ) -> None:
        self._synchronize_device()
        step = int(self._step_getter() if step is None else step)
        expected = _as_signed_int64(self._expected_next_subsequence(step))
        snapshot = (
            self._local_state_matrix().unsqueeze(0).detach().cpu()
            if states is None
            else states.detach().cpu()
        )
        expected_shape = (snapshot.shape[0], len(self.logical_keys), 2)
        if tuple(snapshot.shape) != expected_shape:
            raise RuntimeError(
                f"invalid output-head SR progress matrix {tuple(snapshot.shape)}"
            )
        for rank in range(snapshot.shape[0]):
            actual = int(snapshot[rank, 0, 1].item())
            if actual != expected:
                raise RuntimeError(
                    "output-head SR invocation count disagrees with optimizer "
                    "step/microbatch geometry: "
                    f"rank={rank}, step={step}, origin={self.phase_origin_step}, "
                    f"gradient_accumulation={self.gradient_accumulation_steps}, "
                    f"expected_next={_as_uint64(expected)}, "
                    f"actual_next={_as_uint64(actual)}"
                )

    def state_dict(self) -> dict[str, object]:
        states = self._gather_rank_states()
        self.validate_progress(states=states)
        return {
            "version": torch.tensor([STATE_VERSION], dtype=torch.int64),
            "seed_namespace_version": torch.tensor(
                [SEED_NAMESPACE_VERSION], dtype=torch.int64
            ),
            "seed_base": torch.tensor(
                [_as_signed_int64(self._active_user_seed)], dtype=torch.int64
            ),
            "subsequence_base": torch.tensor(
                [_as_signed_int64(self._active_subsequence_base)], dtype=torch.int64
            ),
            "subsequence_stride": torch.tensor([SUBSEQUENCE_STRIDE], dtype=torch.int64),
            "world_size": torch.tensor([self.world_size], dtype=torch.int64),
            "rank_ids": torch.arange(self.world_size, dtype=torch.int64),
            "gradient_accumulation_steps": torch.tensor(
                [self.gradient_accumulation_steps], dtype=torch.int64
            ),
            "phase_origin_step": torch.tensor(
                [self.phase_origin_step], dtype=torch.int64
            ),
            "logical_keys": list(self.logical_keys),
            "states": states.clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        expected_fields = {
            "version",
            "seed_namespace_version",
            "seed_base",
            "subsequence_base",
            "subsequence_stride",
            "world_size",
            "rank_ids",
            "gradient_accumulation_steps",
            "phase_origin_step",
            "logical_keys",
            "states",
        }
        if set(state_dict) != expected_fields:
            raise RuntimeError(
                "invalid output-head SR checkpoint fields: "
                f"expected={sorted(expected_fields)}, got={sorted(state_dict)}"
            )
        version = _checkpoint_int64_scalar("version", state_dict["version"])
        namespace_version = _checkpoint_int64_scalar(
            "seed_namespace_version", state_dict["seed_namespace_version"]
        )
        stride = _checkpoint_int64_scalar(
            "subsequence_stride", state_dict["subsequence_stride"]
        )
        checkpoint_world = _checkpoint_int64_scalar(
            "world_size", state_dict["world_size"]
        )
        checkpoint_ga = _checkpoint_int64_scalar(
            "gradient_accumulation_steps",
            state_dict["gradient_accumulation_steps"],
        )
        if version != STATE_VERSION:
            raise RuntimeError(
                f"unsupported output-head SR state version {version}; "
                f"expected {STATE_VERSION}"
            )
        if namespace_version != SEED_NAMESPACE_VERSION:
            raise RuntimeError(
                "unsupported output-head SR seed namespace version "
                f"{namespace_version}; expected {SEED_NAMESPACE_VERSION}"
            )
        if stride != SUBSEQUENCE_STRIDE:
            raise RuntimeError(
                f"output-head SR stride mismatch: checkpoint={stride}, "
                f"runtime={SUBSEQUENCE_STRIDE}"
            )
        if checkpoint_world != self.world_size:
            raise RuntimeError(
                "output-head SR world_size differs from runtime; refusing to "
                f"remap rank streams: checkpoint={checkpoint_world}, "
                f"runtime={self.world_size}"
            )
        if checkpoint_ga != self.gradient_accumulation_steps:
            raise RuntimeError(
                "output-head SR gradient-accumulation geometry differs from "
                f"runtime: checkpoint={checkpoint_ga}, "
                f"runtime={self.gradient_accumulation_steps}"
            )
        rank_ids = _checkpoint_int64_tensor("rank_ids", state_dict["rank_ids"])
        if not torch.equal(rank_ids, torch.arange(self.world_size, dtype=torch.int64)):
            raise RuntimeError("output-head SR rank namespace is malformed")
        logical_keys = tuple(state_dict["logical_keys"])
        if logical_keys != self.logical_keys:
            raise RuntimeError(
                "output-head SR logical-producer manifest differs from runtime"
            )

        seed_base = _as_uint64(
            _checkpoint_int64_scalar("seed_base", state_dict["seed_base"])
        )
        subsequence_base = _as_uint64(
            _checkpoint_int64_scalar("subsequence_base", state_dict["subsequence_base"])
        )
        phase_origin_step = _checkpoint_int64_scalar(
            "phase_origin_step", state_dict["phase_origin_step"]
        )
        if phase_origin_step < 0:
            raise RuntimeError("output-head SR phase origin must be non-negative")
        states = _checkpoint_int64_tensor("states", state_dict["states"])
        expected_shape = (self.world_size, len(self.logical_keys), 2)
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                "invalid output-head SR state matrix shape: "
                f"expected={expected_shape}, got={tuple(states.shape)}"
            )
        for rank in range(self.world_size):
            expected_seed = _as_signed_int64(self._seed_for(seed_base, rank, 0))
            if int(states[rank, 0, 0].item()) != expected_seed:
                raise RuntimeError(
                    "output-head SR checkpoint seed namespace is inconsistent "
                    f"at rank={rank}; refusing resume"
                )

        self._active_user_seed = seed_base
        self._active_subsequence_base = subsequence_base
        self.phase_origin_step = phase_origin_step
        self._states[LOGICAL_KEYS[0]].copy_(states[self.rank, 0].to(self.device))
        self._synchronize_device()

    def reset_to_configured_base(self, phase_origin_step: int) -> None:
        phase_origin_step = int(phase_origin_step)
        if phase_origin_step < 0:
            raise ValueError("output-head SR phase origin must be non-negative")
        self._synchronize_device()
        self._active_user_seed = self.user_seed
        self._active_subsequence_base = self.user_subsequence_base
        self.phase_origin_step = phase_origin_step
        self._states[LOGICAL_KEYS[0]].copy_(
            torch.tensor(
                [
                    _as_signed_int64(self._seed_for(self.user_seed, self.rank, 0)),
                    _as_signed_int64(self.user_subsequence_base),
                ],
                dtype=torch.int64,
                device=self.device,
            )
        )
        self._synchronize_device()


def _required_checkpoint_fields() -> set[str]:
    return {
        "version",
        "seed_namespace_version",
        "seed_base",
        "subsequence_base",
        "subsequence_stride",
        "world_size",
        "rank_ids",
        "gradient_accumulation_steps",
        "phase_origin_step",
        "logical_keys",
        "states",
    }


def checkpoint_output_head_sr_schema(checkpoint_id: str) -> str:
    """Classify this exact ABI from DCP metadata without loading tensors."""
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
    if fields != _required_checkpoint_fields() or not isinstance(
        states_metadata, TensorStorageMetadata
    ):
        return "unknown"
    if len(states_metadata.size) != 3:
        return "unknown"
    return "v1"


def _configured_seed() -> int:
    return int(
        os.environ.get(
            "LBT_OUTPUT_HEAD_SR_SEED",
            os.environ.get(
                "FP4_CCE_V4_MXFP8_G_RNG_SEED",
                os.environ.get("FP4_CCE_V4_NVFP4_RNG_SEED", "0"),
            ),
        )
    )


def _configured_subsequence_base() -> int:
    return int(
        os.environ.get(
            "LBT_OUTPUT_HEAD_SR_SUBSEQUENCE_BASE",
            os.environ.get(
                "FP4_CCE_V4_MXFP8_G_RNG_SUBSEQUENCE_BASE",
                os.environ.get("FP4_CCE_V4_NVFP4_RNG_SUBSEQUENCE_BASE", "0"),
            ),
        )
    )


def _validate_supported_route() -> None:
    required_true = (
        "FP4_CCE_V4_NVFP4_G_ROW_DATA_SR",
        "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW",
        "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE",
        "FP4_CCE_V4_MIXED_DW_MXFP8_COLS",
    )
    missing = [name for name in required_true if not _env_flag(name, False)]
    if missing:
        raise RuntimeError(
            "checkpointed output-head SR supports only the fused MXFP8 G-row "
            f"production route; missing required flags: {missing}"
        )
    misleading_or_broad = (
        "FP4_CCE_V4_NVFP4_G_COL_DATA_SR",
        "FP4_CCE_V4_NVFP4_X_COL_DATA_SR",
        "FP4_CCE_V4_NVFP4_DATA_SR",
        "FP4_CCE_V4_NVFP4_USE_STOCHASTIC_ROUNDING",
    )
    enabled = [name for name in misleading_or_broad if _env_flag(name, False)]
    if enabled:
        raise RuntimeError(
            "checkpointed output-head SR does not cover independent column or "
            "broad NVFP4 SR. In the supported MXFP8-column route the X/G "
            f"column flags are inert; disable these flags: {enabled}"
        )


def build_output_head_sr_state_for_trainer(
    *,
    device: torch.device | str,
    training_steps: int,
    gradient_accumulation_steps: int,
    step_getter: Callable[[], int],
) -> OutputHeadSRState | None:
    try:
        from fp4_cce_TK.v4_common import set_checkpointed_output_head_sr_state
    except ImportError:
        if _env_flag("FP4_CCE_V4_CHECKPOINTED_HEAD_SR", False):
            raise RuntimeError(
                "checkpointed output-head SR requested, but fp4_cce_TK is not importable"
            )
        return None

    set_checkpointed_output_head_sr_state(None)
    if not _env_flag("FP4_CCE_V4_CHECKPOINTED_HEAD_SR", False):
        return None
    _validate_supported_route()
    margin = int(
        os.environ.get(
            "LBT_OUTPUT_HEAD_SR_RESERVATION_MARGIN",
            str(DEFAULT_RESERVATION_MARGIN),
        )
    )
    state = OutputHeadSRState(
        device=device,
        user_seed=_configured_seed(),
        user_subsequence_base=_configured_subsequence_base(),
        training_steps=training_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        step_getter=step_getter,
        reservation_margin=margin,
    )
    set_checkpointed_output_head_sr_state(state.get(device=device))
    return state


def register_with_checkpointer(checkpointer, state: OutputHeadSRState, logger) -> None:
    """Register state and reject ambiguous legacy/future resumes by default."""
    if getattr(checkpointer, "ft_manager", None) is not None:
        raise RuntimeError(
            "checkpointed output-head SR is not compatible with TorchFT replica "
            "checkpoints because their state closure omits late custom states"
        )
    if CHECKPOINT_KEY in checkpointer.states:
        raise RuntimeError(f"duplicate checkpoint state {CHECKPOINT_KEY!r}")
    checkpointer.states[CHECKPOINT_KEY] = state
    original_dcp_load = checkpointer.dcp_load

    def dcp_load_with_output_head_sr(
        this,
        state_dict,
        checkpoint_id,
        from_hf,
        from_quantized,
    ):
        load_state = state_dict
        starts_new_phase = False
        if CHECKPOINT_KEY in state_dict:
            if from_hf:
                schema = "missing"
            else:
                try:
                    schema = checkpoint_output_head_sr_schema(checkpoint_id)
                except Exception as exc:
                    raise RuntimeError(
                        "could not verify output-head SR checkpoint metadata; "
                        "refusing an ambiguous resume"
                    ) from exc
            if schema == "missing":
                policy = os.environ.get(MISSING_POLICY_ENV, "").strip().lower()
                if policy != START_NEW_PHASE_POLICY:
                    raise RuntimeError(
                        "checkpoint predates checkpointed output-head SR. Resume "
                        "is rejected by default because the process-global "
                        "invocation counter cannot be recovered. For an exact "
                        "paired experiment only, explicitly set "
                        f"{MISSING_POLICY_ENV}={START_NEW_PHASE_POLICY} to begin "
                        "a declared new stochastic phase from the configured "
                        "seed/subsequence; no step-based migration is inferred."
                    )
                load_state = dict(state_dict)
                load_state.pop(CHECKPOINT_KEY)
                starts_new_phase = True
            elif schema != "v1":
                raise RuntimeError(
                    "unrecognized or partial output-head SR checkpoint ABI; "
                    "refusing resume"
                )

        result = original_dcp_load(
            load_state,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        if starts_new_phase:
            origin = int(state._step_getter())
            state.reset_to_configured_base(origin)
            logger.warning(
                "Checkpoint predates checkpointed output-head SR; explicitly "
                "started a new rank-namespaced stochastic phase at step %d. "
                "This is not bitwise-continuous with the legacy process-global "
                "counter and no step-based counter inference was performed.",
                origin,
            )
        else:
            state.validate_progress()
        return result

    checkpointer.dcp_load = types.MethodType(dcp_load_with_output_head_sr, checkpointer)
