"""One-GPU fused-head SR uninterrupted-versus-resume proof.

Run only with one verified-idle physical GPU exposed through
``CUDA_VISIBLE_DEVICES``.  The script compiles the pinned FP4 helper, reserves
two stochastic invocations, restores the checkpoint after the first, and
requires the resumed stochastic payload, per-row outputs, and next state to be
bitwise identical.  The scalar loss is a floating-atomic reduction, so it is
checked with a tight numerical tolerance instead of a bitwise assertion.
"""

from __future__ import annotations

import importlib
import json
import os

import torch

from low_bits_training.cce import backend as cce_backend
from low_bits_training.cce.head_sr_state import OutputHeadSRState
from fp4_cce_TK import v4_common as installed_v4_common


EXACT_RESUME_FIELDS = tuple(range(1, 9))
SR_PAYLOAD_FIELDS = (5,)
SCALAR_LOSS_RTOL = 1.0e-6
SCALAR_LOSS_ATOL = 1.0e-6


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()


def _differing_tensor_fields(
    left: tuple,
    right: tuple,
    fields: tuple[int, ...] | None = None,
) -> list[int]:
    selected = range(min(len(left), len(right))) if fields is None else fields
    return [
        index
        for index in selected
        for actual, expected in ((left[index], right[index]),)
        if torch.is_tensor(actual)
        and torch.is_tensor(expected)
        and not torch.equal(_bytes(actual), _bytes(expected))
    ]


def _assert_resume_equivalent(uninterrupted: tuple, resumed: tuple) -> None:
    differing_exact_fields = _differing_tensor_fields(
        uninterrupted,
        resumed,
        EXACT_RESUME_FIELDS,
    )
    if differing_exact_fields:
        raise RuntimeError(
            "resumed fused-head exact fields are not bitwise identical: "
            f"{differing_exact_fields}"
        )
    try:
        torch.testing.assert_close(
            uninterrupted[0],
            resumed[0],
            rtol=SCALAR_LOSS_RTOL,
            atol=SCALAR_LOSS_ATOL,
        )
    except AssertionError as exc:
        loss_delta = abs(float(uninterrupted[0]) - float(resumed[0]))
        raise RuntimeError(
            "resumed fused-head scalar loss exceeds the floating-reduction "
            f"tolerance: absolute_delta={loss_delta:.17g}, "
            f"rtol={SCALAR_LOSS_RTOL}, atol={SCALAR_LOSS_ATOL}"
        ) from exc


def _assert_result_contract(
    label: str, result: tuple, reference: tuple | None = None
) -> None:
    if len(result) != 9:
        raise RuntimeError(f"{label} returned {len(result)} fields, expected 9")
    if reference is not None:
        for index, (actual, expected) in enumerate(zip(result, reference)):
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.device != expected.device
            ):
                raise RuntimeError(
                    f"{label} field {index} contract mismatch: "
                    f"actual=({actual.shape}, {actual.dtype}, {actual.device}) "
                    f"expected=({expected.shape}, {expected.dtype}, {expected.device})"
                )
    # Fields 5 and 6 contain opaque packed row values and scale codes; their
    # storage bit patterns are not floating-point semantic values.
    for index in (0, 1, 2, 3, 8):
        if not torch.isfinite(result[index].float()).all():
            raise RuntimeError(
                f"{label} semantic field {index} contains a nonfinite value"
            )


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") != 0:
        raise RuntimeError("expose exactly one verified-idle GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    os.environ["FP4_CCE_V4_CHECKPOINTED_HEAD_SR"] = "1"
    os.environ["FP4_CCE_V4_NVFP4_G_ROW_DATA_SR"] = "1"
    os.environ["FP4_CCE_V4_NVFP4_DATA_SR"] = "0"
    os.environ["FP4_CCE_V4_NVFP4_USE_STOCHASTIC_ROUNDING"] = "0"

    step = [0]
    state = OutputHeadSRState(
        device=device,
        user_seed=42,
        user_subsequence_base=17,
        training_steps=4,
        gradient_accumulation_steps=1,
        step_getter=lambda: step[0],
        reservation_margin=2,
    )
    owner_tensor = state.get(device=device)
    installed_v4_common.set_checkpointed_output_head_sr_state(owner_tensor)

    # Match production ordering exactly: trainer state installation happens
    # before the backend's first lazy runtime load.  That load must preserve
    # the module owning the installed tensor rather than silently re-importing
    # v4_common with an empty module global.
    cce_backend._load_fp4_cce_tk_v4()
    v4_common = importlib.import_module("fp4_cce_TK.v4_common")
    if v4_common is not installed_v4_common:
        raise RuntimeError(
            "production lazy loader replaced the output-head SR owner module"
        )
    if (
        v4_common._checkpointed_output_head_sr_state_for(owner_tensor)
        is not owner_tensor
    ):
        raise RuntimeError(
            "production runtime did not retain the checkpoint-owned SR tensor"
        )

    generator = torch.Generator(device=device).manual_seed(20260821)
    rows = cols = hidden = 128
    x = (
        torch.randn(
            (rows, hidden), device=device, dtype=torch.bfloat16, generator=generator
        )
        * 0.125
    )
    weight = (
        torch.randn(
            (cols, hidden), device=device, dtype=torch.bfloat16, generator=generator
        )
        * 0.125
    )
    # The exact-selected-logit repair assumes the supplied logits came from
    # this x/weight pair.  Keep the proof fixture consistent with production.
    logits = torch.mm(x.float(), weight.float().t()).to(torch.bfloat16).contiguous()
    targets = torch.arange(rows, device=device, dtype=torch.int64) % cols
    valid = torch.ones(rows, device=device, dtype=torch.bool)

    def produce() -> tuple:
        return v4_common.direct_loss_lse_target_topk_split_exact_logits_mxfp8_row(
            # The fused producer consumes selected entries from logits in
            # place.  Each logical invocation receives a fresh activation.
            logits.clone(),
            x,
            weight,
            targets,
            valid,
            cols,
            0,
            448.0,
            False,
        )

    first = produce()
    _assert_result_contract("explicit state path", first)
    step[0] = 1
    checkpoint = state.state_dict()
    second_uninterrupted = produce()
    _assert_result_contract("uninterrupted second output", second_uninterrupted, first)
    step[0] = 2
    state.validate_progress()
    uninterrupted_next_state = state.get().detach().clone()

    resumed_step = [1]
    resumed = OutputHeadSRState(
        device=device,
        user_seed=42,
        user_subsequence_base=17,
        training_steps=4,
        gradient_accumulation_steps=1,
        step_getter=lambda: resumed_step[0],
        reservation_margin=2,
    )
    resumed.load_state_dict(checkpoint)
    resumed.validate_progress()
    v4_common.set_checkpointed_output_head_sr_state(resumed.get(device=device))
    second_resumed = produce()
    _assert_result_contract("resumed second output", second_resumed, first)
    resumed_step[0] = 2
    resumed.validate_progress()

    if not _differing_tensor_fields(
        first,
        second_uninterrupted,
        SR_PAYLOAD_FIELDS,
    ):
        raise RuntimeError(
            "SR payload did not advance between adjacent invocations"
        )
    _assert_resume_equivalent(second_uninterrupted, second_resumed)
    if not torch.equal(uninterrupted_next_state, resumed.get()):
        raise RuntimeError("resumed fused-head SR next state is not bitwise exact")

    v4_common.set_checkpointed_output_head_sr_state(None)
    os.environ["FP4_CCE_V4_CHECKPOINTED_HEAD_SR"] = "0"
    legacy_default = produce()
    _assert_result_contract("legacy default path", legacy_default, first)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "first_sr_payload_differs_from_second": True,
                "legacy_default_path_executes": True,
                "production_lazy_loader_preserves_module_identity": True,
                "production_runtime_uses_owner_state": True,
                "resumed_nonreduction_fields_bitwise_exact": True,
                "resumed_next_state_bitwise_exact": True,
                "resumed_scalar_loss_close": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
