#!/usr/bin/env python3
"""Exact eager WO equivalence/lifetime gate for the guarded RHT overlap.

This gate compares the production localCTA-v4 WO consumer with
``USE_TK_LOCALCTA_V4_WO_RHT_WEIGHT_QUANT_OVERLAP`` disabled and enabled in one
process.  Every arm starts from identical BF16 tensors, PyTorch RNG state, and
checkpoint-style WO gradient-SR state.  It checks the exact WO output, dInput,
and dWeight bytes and the exact single ``2**32`` SR reservation.

The overlap arms also stress the two ownership transfers made by the candidate:

* the transient BF16 weight is consumed by the weight side stream; and
* the quantized weight carrier is produced there but consumed on the caller
  stream, including by backward.

Artificial delays keep both uses live while transient sources and the autograd
context are dropped.  A third stream then allocates and writes tensors matching
the protected source/carrier allocations.  Any premature pointer reuse or
result corruption fails the gate.  CUDA graphs remain rejected by the product
path and are deliberately not tested here.

Run after sourcing the production localCTA-v4 row-SR/column-RHT profile::

    PYTHONPATH="$PWD:$TORCHTITAN_ROOT:$FP4_OVERLAY" \
      python tools/check_localcta_rht_wo_overlap_equivalence.py \
        --device cuda:0 --scale-num 448 \
        --expected-extension /absolute/path/to/_tk_quant_localcta_v4.so

The script is synthetic and therefore proves same-process implementation
equivalence and allocator safety, not checkpoint provenance.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import weakref

import torch


SELECTOR_ENV = "USE_TK_LOCALCTA_V4_WO_RHT_WEIGHT_QUANT_OVERLAP"
SUBSEQUENCE_STRIDE = 1 << 32
UINT64_MASK = (1 << 64) - 1
RESULT_NAMES = ("output", "grad_input", "grad_weight")
MEASURED_ARMS = ("sequential", "overlap", "overlap")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--gate-seed", type=int, default=20260828)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--rng-subsequence", type=int, default=17)
    parser.add_argument("--scale-num", type=float, required=True)
    parser.add_argument("--expected-extension", required=True)
    parser.add_argument(
        "--sleep-cycles",
        type=int,
        default=200_000_000,
        help="CUDA clock cycles used to keep source/carrier uses in flight",
    )
    parser.add_argument(
        "--churn-repeats",
        type=int,
        default=2,
        help="Matching allocations written on the independent churn stream",
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.rows % 256:
        parser.error("--rows must be a positive multiple of 256")
    if args.dim <= 0 or args.dim % 256:
        parser.error("--dim must be a positive multiple of 256")
    if args.scale_num != 448.0:
        parser.error("this production gate requires --scale-num 448")
    if args.sleep_cycles <= 0:
        parser.error("--sleep-cycles must be positive; lifetime stress is mandatory")
    if args.churn_repeats <= 0:
        parser.error("--churn-repeats must be positive")
    return args


def _configure(args: argparse.Namespace) -> None:
    """Seal the production WO policy before importing route-sensitive LBT."""
    values = {
        "USE_TK_QUANT": "1",
        "USE_TK_GEMM": "1",
        "USE_TK_LOCALCTA": "1",
        "USE_TK_LOCALCTA_VARIANT": "v4",
        "USE_TK_LOCALCTA_FORWARD_MIN_M": "256",
        "USE_TK_LOCALCTA_SCALE_NUM": "448",
        "USE_TK_LOCALCTA_2D_WEIGHT_QUANT": "1",
        "USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER": "0",
        "USE_TK_LOCALCTA_V4_ROW_PREPARED_COL_OUTER": "1",
        "USE_TK_LOCALCTA_V4_FAST_FORWARD_GEMM": "1",
        "USE_TK_LOCALCTA_V4_FAST_WO_DGRAD": "1",
        "USE_TK_LOCALCTA_V4_FAST_WO_WGRAD": "1",
        "USE_TK_LOCALCTA_V4_WO_BF16_BWD": "0",
        "USE_TK_LOCALCTA_PAIRED_RHT_CARRIER": "1",
        "USE_TK_WO_ROWONLY_INPUT_QUANT": "0",
        "USE_TK_MS": "0",
        "USE_CUDA_GRAPH": "0",
        "NVTE_NVFP4_ENCODE_CENTRIC": "0",
        "NVFP4_USE_RHT": "1",
        "NVFP4_RHT_AXES": "col",
        "NVFP4_RHT_RANDOM_SIGNS": "1",
        "NVFP4_RHT_ACTIVATION": "1",
        "NVFP4_RHT_GRAD": "1",
        "NVFP4_RHT_WEIGHT": "0",
        "NVFP4_USE_STOCHASTIC_ROUNDING": "1",
        "NVFP4_SR_ACTIVATION": "0",
        "NVFP4_SR_GRAD": "1",
        "NVFP4_GRAD_SR_AXES": "row",
        "NVFP4_SR_WEIGHT": "0",
        "NVFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SCALE_SR_ACTIVATION": "0",
        "NVFP4_SCALE_SR_GRAD": "0",
        "NVFP4_SCALE_SR_WEIGHT": "0",
        "NVFP4_RNG_SEED": str(args.rng_seed),
        "NVFP4_RNG_SUBSEQUENCE_BASE": str(args.rng_subsequence),
    }
    os.environ.update(values)


def _as_u64(value: int) -> int:
    return int(value) & UINT64_MASK


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{name}: metadata differs: actual={actual.shape}/{actual.dtype}, "
            f"expected={expected.shape}/{expected.dtype}"
        )
    a = _byte_view(actual)
    b = _byte_view(expected)
    if torch.equal(a, b):
        return
    mismatches = int((a != b).sum().item())
    max_abs = None
    if actual.is_floating_point():
        max_abs = float((actual.float() - expected.float()).abs().max().item())
    raise AssertionError(
        f"{name}: {mismatches}/{a.numel()} bytes differ; max_abs={max_abs}"
    )


def _descriptor(name: str, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "name": name,
        "ptr": int(tensor.data_ptr()),
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "nbytes": int(tensor.numel() * tensor.element_size()),
    }


def _carrier_descriptors(quantized) -> list[dict[str, object]]:
    fields = {
        "row": quantized._tk_row,
        "col": quantized._tk_col,
        "row_chunk_sg": quantized._tk_row_chunk_sg,
        "col_chunk_sg": quantized._tk_col_chunk_sg,
        "keepalive": quantized._keepalive,
    }
    descriptors: list[dict[str, object]] = []
    seen: set[int] = set()

    def visit(prefix: str, value) -> None:
        if torch.is_tensor(value):
            if not value.is_cuda or value.numel() == 0:
                return
            ptr = int(value.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                descriptors.append(_descriptor(prefix, value))
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(f"{prefix}[{index}]", item)

    for name, value in fields.items():
        visit(name, value)
    if not descriptors:
        raise RuntimeError("weight quantization returned no live CUDA carrier tensors")
    return descriptors


def _snapshot_sr(state, key: str) -> torch.Tensor:
    return state.get(key).detach().clone()


def _restore_sr(state, key: str, initial: torch.Tensor) -> None:
    state.get(key).copy_(initial)


def _assert_one_reservation(
    name: str, before: torch.Tensor, after: torch.Tensor
) -> None:
    before_cpu = before.detach().cpu()
    after_cpu = after.detach().cpu()
    if int(after_cpu[0]) != int(before_cpu[0]):
        raise AssertionError(f"{name}: Philox seed changed")
    expected = (_as_u64(int(before_cpu[1])) + SUBSEQUENCE_STRIDE) & UINT64_MASK
    actual = _as_u64(int(after_cpu[1]))
    if actual != expected:
        raise AssertionError(
            f"{name}: expected one +2^32 reservation, got "
            f"before={_as_u64(int(before_cpu[1]))}, after={actual}"
        )


def _allocate_churn(
    descriptors: list[dict[str, object]],
    *,
    stream: torch.cuda.Stream,
    device: torch.device,
    repeats: int,
) -> tuple[list[torch.Tensor], set[int], int]:
    allocations: list[torch.Tensor] = []
    pointers: set[int] = set()
    total_bytes = 0
    with torch.cuda.stream(stream):
        for repeat in range(repeats):
            for desc in descriptors:
                value = torch.empty(
                    desc["shape"], dtype=desc["dtype"], device=device
                )
                value.view(torch.uint8).fill_(0xA5 if repeat % 2 == 0 else 0x3C)
                allocations.append(value)
                pointers.add(int(value.data_ptr()))
                total_bytes += int(desc["nbytes"])
    return allocations, pointers, total_bytes


def _run_arm(
    *,
    route: str,
    fte,
    input_source: torch.Tensor,
    weight_source: torch.Tensor,
    upstream_source: torch.Tensor,
    quantizers: tuple[object, object, object],
    workspace: torch.Tensor,
    debug_name: str,
    sr_state,
    sr_key: str,
    initial_sr: torch.Tensor,
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    device: torch.device,
    sleep_cycles: int,
    churn_repeats: int,
) -> dict[str, object]:
    overlap = route == "overlap"
    os.environ[SELECTOR_ENV] = "1" if overlap else "0"
    if fte.use_tk_localcta_v4_wo_rht_weight_quant_overlap() != overlap:
        raise RuntimeError(f"failed to select requested WO route {route}")
    if fte.use_cuda_graph():
        raise RuntimeError("WO overlap equivalence is eager-only; CUDA graph is active")
    if fte._nvfp4_quantizer_extras_enabled("weight"):
        raise RuntimeError("weight SR/RHT extras make WO overlap unsupported")

    torch.cuda.synchronize(device)
    _restore_sr(sr_state, sr_key, initial_sr)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device=device)
    torch.cuda.synchronize(device)

    caller = torch.cuda.current_stream(device)
    side = fte._get_ms_stream()
    churn_stream = torch.cuda.Stream(device=device)
    if overlap:
        if not hasattr(torch.cuda, "_sleep"):
            raise RuntimeError("torch.cuda._sleep is required for lifetime stress")
        with torch.cuda.stream(side):
            torch.cuda._sleep(sleep_cycles)

    traces: list[dict[str, object]] = []
    original_fast_quantize = fte._fast_quantize

    def audited_fast_quantize(tensor, quantizer=None, *args, **kwargs):
        role = fte._nvfp4_quantizer_role(
            quantizer, kwargs.get("nvfp4_role")
        )
        result = original_fast_quantize(tensor, quantizer, *args, **kwargs)
        trace: dict[str, object] = {
            "role": role,
            "stream": int(torch.cuda.current_stream(device).cuda_stream),
            "input_ptr": int(tensor.data_ptr()),
        }
        if role == "weight":
            trace["carrier"] = _carrier_descriptors(result)
        traces.append(trace)
        return result

    fte._fast_quantize = audited_fast_quantize
    try:
        x = input_source.detach().clone().requires_grad_(True)
        weight = weight_source.detach().clone().requires_grad_(True)
        upstream = upstream_source.detach().clone()
        source_descriptors = [_descriptor("bf16_weight", weight)]
        source_ptrs = {int(weight.data_ptr())}
        x_ref = weakref.ref(x)
        weight_ref = weakref.ref(weight)

        empty = torch.empty(0, dtype=torch.bfloat16, device=device)
        output = fte._WoFunction_TK.apply(
            x,
            weight,
            quantizers[0],
            quantizers[1],
            quantizers[2],
            workspace,
            debug_name,
            empty,
            empty,
            False,
        )
        after_forward_sr = _snapshot_sr(sr_state, sr_key)
        if overlap:
            # Keep the side-produced carrier live in the caller's backward
            # while the independent churn stream tries matching allocations.
            torch.cuda._sleep(sleep_cycles)
        grad_input, grad_weight = torch.autograd.grad(
            output, (x, weight), grad_outputs=upstream
        )

        snapshots = {
            "output": output.detach().clone(),
            "grad_input": grad_input.detach().clone(),
            "grad_weight": grad_weight.detach().clone(),
        }
        input_after = x.detach().clone()
        weight_after = weight.detach().clone()

        weight_traces = [trace for trace in traces if trace["role"] == "weight"]
        activation_traces = [
            trace for trace in traces if trace["role"] == "activation"
        ]
        grad_traces = [trace for trace in traces if trace["role"] == "grad"]
        if len(weight_traces) != 1 or len(activation_traces) != 1 or len(grad_traces) != 1:
            raise RuntimeError(
                "expected exactly one activation/weight/grad quantization; got "
                f"activation={len(activation_traces)}, weight={len(weight_traces)}, "
                f"grad={len(grad_traces)}"
            )
        carrier_descriptors = weight_traces[0]["carrier"]
        carrier_ptrs = {int(desc["ptr"]) for desc in carrier_descriptors}
        if source_ptrs & carrier_ptrs:
            raise AssertionError("quantized weight carrier aliases its BF16 source")

        caller_id = int(caller.cuda_stream)
        activation_stream = int(activation_traces[0]["stream"])
        weight_stream = int(weight_traces[0]["stream"])
        if activation_stream != caller_id:
            raise AssertionError("activation quantization left the caller stream")
        if overlap and weight_stream == caller_id:
            raise AssertionError("overlap selector did not move weight quantization")
        if not overlap and weight_stream != caller_id:
            raise AssertionError("sequential selector unexpectedly used a side stream")

        del output, grad_input, grad_weight, upstream, x, weight, empty
        gc.collect()
        if x_ref() is not None or weight_ref() is not None:
            raise AssertionError(
                "transient BF16 sources remain Python-referenced after backward; "
                "lifetime stress would be inconclusive"
            )

        protected = source_descriptors + carrier_descriptors
        churn, churn_ptrs, churn_bytes = _allocate_churn(
            protected,
            stream=churn_stream,
            device=device,
            repeats=churn_repeats,
        )
        premature_aliases = (source_ptrs | carrier_ptrs) & churn_ptrs
        torch.cuda.synchronize(device)
        if premature_aliases:
            raise AssertionError(
                "allocator reused protected in-flight pointers during churn: "
                + ", ".join(hex(ptr) for ptr in sorted(premature_aliases))
            )
        del churn

        after_sr = _snapshot_sr(sr_state, sr_key)
        after_cpu_rng = torch.get_rng_state().clone()
        after_cuda_rng = torch.cuda.get_rng_state(device).clone()
        _assert_exact(f"{route}.forward_sr_unchanged", after_forward_sr, initial_sr)
        _assert_one_reservation(f"{route}.wo_grad_sr", initial_sr, after_sr)
        _assert_exact(f"{route}.cpu_rng", after_cpu_rng, cpu_rng)
        _assert_exact(f"{route}.cuda_rng", after_cuda_rng, cuda_rng)
        _assert_exact(f"{route}.input_unchanged", input_after, input_source)
        _assert_exact(f"{route}.weight_unchanged", weight_after, weight_source)
        for name, value in snapshots.items():
            if not bool(torch.isfinite(value).all()):
                raise AssertionError(f"{route}.{name} contains non-finite values")
        return {
            "route": route,
            "snapshots": snapshots,
            "after_sr": after_sr,
            "streams": {
                "caller": caller_id,
                "activation_quant": activation_stream,
                "weight_quant": weight_stream,
            },
            "pointers": {
                "source": sorted(source_ptrs),
                "weight_carrier": sorted(carrier_ptrs),
            },
            "churn_bytes": churn_bytes,
        }
    finally:
        fte._fast_quantize = original_fast_quantize


def main() -> None:
    args = _parse_args()
    _configure(args)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device required, got {device}")
    if device.index is not None:
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda", torch.cuda.current_device())

    from low_bits_training.quantization import fused_te_linear as fte
    from low_bits_training.quantization import tk_gemm
    from low_bits_training.quantization.localcta_sr_state import (
        LocalCTASRState,
        active_localcta_sr_state,
        set_active_localcta_sr_state,
        wo_grad_key,
    )

    if fte.use_cuda_graph():
        raise RuntimeError("CUDA graphs must be disabled for the overlap candidate")
    if not fte.use_tk_localcta() or tk_gemm.get_tk_localcta_variant() != "v4":
        raise RuntimeError("gate requires localCTA v4")
    if not fte.use_nvfp4_rht_for_role("activation"):
        raise RuntimeError("gate requires activation column RHT")
    if fte._nvfp4_native_rht_axes_for_role("activation") != "col":
        raise RuntimeError("gate requires column-only RHT")
    if not fte.use_nvfp4_data_stochastic_rounding_for_role("grad"):
        raise RuntimeError("gate requires gradient row-SR")
    if fte._nvfp4_grad_sr_axes() != "row":
        raise RuntimeError("gate requires row-only gradient SR")

    tkq = tk_gemm._get_tk_quant_for_gemm()
    raw_module = getattr(tkq, "_mod", None)
    if raw_module is None or not getattr(raw_module, "__file__", None):
        raise RuntimeError("could not resolve raw localCTA quant extension")
    actual_extension = Path(raw_module.__file__).resolve()
    expected_extension = Path(args.expected_extension).resolve()
    if actual_extension != expected_extension:
        raise RuntimeError(
            f"stale/wrong extension: loaded {actual_extension}; "
            f"expected {expected_extension}"
        )
    if not hasattr(tkq, "tk_set_global_scale_num") or not hasattr(
        tkq, "tk_get_global_scale_num"
    ):
        raise RuntimeError("localCTA extension lacks global scale controls")

    previous_scale_num = float(tkq.tk_get_global_scale_num())
    previous_active_sr = active_localcta_sr_state()
    previous_selector = os.environ.get(SELECTOR_ENV)
    debug_name = "wo_overlap_gate.layers.0.attention:wo"
    sr_key = wo_grad_key(debug_name)
    try:
        tkq.tk_set_global_scale_num(args.scale_num)
        if float(tkq.tk_get_global_scale_num()) != args.scale_num:
            raise RuntimeError("localCTA extension rejected scale numerator 448")

        sr_state = LocalCTASRState(
            (sr_key,),
            device=device,
            user_seed=args.rng_seed,
            user_subsequence_base=args.rng_subsequence,
            training_steps=1,
            gradient_accumulation_steps=1,
            rank=0,
            world_size=1,
        )
        set_active_localcta_sr_state(sr_state)

        torch.manual_seed(args.gate_seed)
        torch.cuda.manual_seed_all(args.gate_seed)
        input_source = torch.randn(
            args.rows, args.dim, dtype=torch.bfloat16, device=device
        ).mul_(0.5).contiguous()
        weight_source = torch.randn(
            args.dim, args.dim, dtype=torch.bfloat16, device=device
        ).mul_(0.02).contiguous()
        upstream_source = torch.randn(
            args.rows, args.dim, dtype=torch.bfloat16, device=device
        ).mul_(0.01).contiguous()
        quantizers = (
            fte._make_nvfp4_quantizer_for_role("activation"),
            fte._make_nvfp4_quantizer_for_role("weight"),
            fte._make_nvfp4_quantizer_for_role("grad"),
        )
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        initial_sr = _snapshot_sr(sr_state, sr_key)

        fte.clear_fused_fp4_step_caches()
        reference: dict[str, torch.Tensor] | None = None
        reference_sr: torch.Tensor | None = None
        arm_reports: list[dict[str, object]] = []
        for index, route in enumerate(MEASURED_ARMS):
            arm = _run_arm(
                route=route,
                fte=fte,
                input_source=input_source,
                weight_source=weight_source,
                upstream_source=upstream_source,
                quantizers=quantizers,
                workspace=workspace,
                debug_name=debug_name,
                sr_state=sr_state,
                sr_key=sr_key,
                initial_sr=initial_sr,
                cpu_rng=cpu_rng,
                cuda_rng=cuda_rng,
                device=device,
                sleep_cycles=args.sleep_cycles,
                churn_repeats=args.churn_repeats,
            )
            snapshots = arm.pop("snapshots")
            after_sr = arm.pop("after_sr")
            if reference is None:
                reference = snapshots
                reference_sr = after_sr
            else:
                for name in RESULT_NAMES:
                    _assert_exact(
                        f"sequential_vs_{route}[{index}].{name}",
                        snapshots[name],
                        reference[name],
                    )
                assert reference_sr is not None
                _assert_exact(
                    f"sequential_vs_{route}[{index}].sr", after_sr, reference_sr
                )
                del snapshots
            arm_reports.append({"index": index, **arm})

        print(
            json.dumps(
                {
                    "status": "PASS_SYNTHETIC_WO_OVERLAP_EQUIVALENCE",
                    "checkpoint_locked": False,
                    "gate": "localcta_v4_wo_rht_weight_overlap_consumer_equivalence",
                    "extension": str(actual_extension),
                    "rows": args.rows,
                    "dim": args.dim,
                    "dtype": "torch.bfloat16",
                    "scale_num": args.scale_num,
                    "rht": "column activation+gradient, fixed signs",
                    "sr": "row-only gradient data-SR, one explicit reservation",
                    "measured_order": list(MEASURED_ARMS),
                    "sleep_cycles": args.sleep_cycles,
                    "churn_repeats": args.churn_repeats,
                    "arms": arm_reports,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        torch.cuda.synchronize(device)
        tkq.tk_set_global_scale_num(previous_scale_num)
        set_active_localcta_sr_state(previous_active_sr)
        if previous_selector is None:
            os.environ.pop(SELECTOR_ENV, None)
        else:
            os.environ[SELECTOR_ENV] = previous_selector


if __name__ == "__main__":
    main()
