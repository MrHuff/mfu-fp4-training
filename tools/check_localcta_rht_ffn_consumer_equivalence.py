#!/usr/bin/env python3
r"""Bit-exact native/fallback equivalence gate through a real production FFN.

This is the consumer-level companion to
``check_localcta_rht_split2_equivalence.py``.  The lower-level gate proves the
carrier payloads themselves.  This gate keeps one real
``FusedFeedForwardFP4_TK`` and its persistent caches alive in one process, then
runs the production downstream GEMM/RMSNorm consumers with the native split2
carrier enabled and disabled.  It therefore tests pointer/lifetime/order effects
without treating ordinary cross-process trainer drift as a native/fallback
difference.

Inputs
------

Outside the explicitly non-checkpoint-locked ``--synthetic`` mode,
``--io-state`` is mandatory and must be produced by
``tools/capture_ffn_checkpoint_io.py``.  It contains two 2-D BF16-compatible
tensors, ``input`` and ``upstream``, plus the captured layer/module identity.
The capture and this gate must use the *same* model checkpoint and config; the
current capture format cannot verify those paths for us.

The FFN weights come from exactly one of:

* ``--config CONFIG --checkpoint CHECKPOINT`` (preferred): load the converted
  checkpoint with ``Generator`` and retain only the selected real FFN; or
* ``--ffn-state STATE``: a torch file with ``norm_weight``, ``w1_weight``,
  ``w3_weight``, and ``w2_weight`` tensors.  This is compatible with the FFN
  state format accepted by ``tests/diagnose_llama8b_fp4_body.py``.

By default the gate constructs a deterministic rank-0 SR state from
``--rng-seed``/``--rng-subsequence``.  For a checkpoint-locked test, pass
``--sr-state`` containing the exact two signed int64[2] CUDA-state values saved
as CPU tensors under ``ffn_w2_grad`` and ``ffn_deriv_grad``.  Element zero is
the Philox seed and element one is the next subsequence.  A compact file can be
created at a trusted checkpoint/capture boundary with::

    torch.save({
        "ffn_w2_grad": state.get(ffn_w2_grad_key(debug_name)).cpu(),
        "ffn_deriv_grad": state.get(ffn_deriv_grad_key(debug_name)).cpu(),
    }, output)

Exact invocation (after sourcing the same production localCTA-v4 RHT profile)
looks like::

    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    python tools/capture_ffn_checkpoint_io.py \
      --config CONFIG --checkpoint CHECKPOINT --output /tmp/ffn-io.pt \
      --layer 31 --seq-len 8192 --batch-size 4 --device cuda:0
    python tools/check_localcta_rht_ffn_consumer_equivalence.py \
      --config CONFIG --checkpoint CHECKPOINT --io-state /tmp/ffn-io.pt \
      --layer 31 --device cuda:0 --scale-num 448 \
      --expected-extension /absolute/path/to/_tk_quant_localcta_v4.so

Add ``--sr-state /tmp/ffn-sr.pt`` to make the RNG point checkpoint-locked.

When no checkpoint/capture is locally available, ``--synthetic`` runs the
same production FFN functions at the fixed Llama-8B shape
``M=32768, K=4096, H=14336`` with seed-initialized BF16 weights, input,
upstream, and explicit persistent SR tensors::

    python tools/check_localcta_rht_ffn_consumer_equivalence.py \
      --synthetic --device cuda:0 --scale-num 448 \
      --expected-extension /absolute/path/to/_tk_quant_localcta_v4.so

Synthetic success closes only in-process pointer/GEMM equivalence.  It is
always emitted as ``PASS_SYNTHETIC_EQUIVALENCE_NOT_CHECKPOINT_LOCKED`` with a
``FAIL_SYNTHETIC_NOT_CHECKPOINT_LOCKED`` checkpoint-lock label.  It cannot
promote a checkpoint continuation or replace the captured-input gate above.

The production profile must select localCTA v4, eager execution, the paired
column-RHT carrier (activation+gradient RHT, no weight RHT), fixed signs,
row-only gradient data SR, no activation/weight data SR, no scale SR, and
encode-centric=false.  The tool validates those requirements rather than
silently replacing the caller's recipe.

The measured order is N/N/N/F/F/N.  Every arm starts from identical CPU/CUDA
RNG and persistent SR state, while FFN/TK caches remain live.  The first native
arm is the reference.  Output, grad_input, grad_w1, grad_w3, grad_w2, and both
SR states remain byte-exact requirements.  ``grad_norm_weight`` is produced by
an independently scheduled BF16 RMSNorm reduction and is not intrinsically
bitwise repeatable.  The tool therefore reports every native/native pair and
each fallback/reference comparison, builds a component-wise conservative
native envelope, and requires every fallback metric to remain inside it.  Both
envelopes must also stay below an absolute max-difference cap of ``2^-10``
(``0.0009765625``), twice the worst native-repeat deviation observed while
developing this gate.  No other exact comparison is relaxed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


SELECTOR_ENV = "USE_TK_LOCALCTA_NATIVE_PAIRED_RHT_SPLIT2"
PRODUCTION_SCALE_NUM = 448.0
MEASURED_ROUTES = (
    "native",
    "native",
    "native",
    "fallback",
    "fallback",
    "native",
)
RESULT_NAMES = (
    "output",
    "grad_input",
    "grad_norm_weight",
    "grad_w1",
    "grad_w3",
    "grad_w2",
)
EXACT_RESULT_NAMES = tuple(name for name in RESULT_NAMES if name != "grad_norm_weight")
COMPACT_SR_KEYS = ("ffn_w2_grad", "ffn_deriv_grad")
UINT64_MASK = (1 << 64) - 1
SUBSEQUENCE_STRIDE = 1 << 32
GRAD_NORM_WEIGHT_ABS_CAP = 2.0**-10
GRAD_NORM_ENVELOPE_METRICS = (
    "mismatched_elements",
    "max_abs",
    "mean_abs",
    "rmse",
)
SYNTHETIC_ROWS = 32768
SYNTHETIC_DIM = 4096
SYNTHETIC_HIDDEN_DIM = 14336
SYNTHETIC_LAYER = 31
SYNTHETIC_UPSTREAM_SCALE = 0.01
SYNTHETIC_SR_SEED = 42
SYNTHETIC_SR_SUBSEQUENCE = 17


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Diagnostic-only fixed production-shape tensors; explicitly NOT "
            "checkpoint-locked"
        ),
    )
    parser.add_argument(
        "--io-state",
        help="Payload written by tools/capture_ffn_checkpoint_io.py",
    )
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--checkpoint",
        help="Converted model checkpoint loaded with Generator (preferred)",
    )
    weights.add_argument(
        "--ffn-state",
        help="Compact real-FFN weight payload; see module documentation",
    )
    parser.add_argument(
        "--config",
        help="Model config; required with --checkpoint and rejected otherwise",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Converted FFN layer (default: layer recorded in --io-state)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Leading captured rows to exercise (default: all captured rows)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generator load seed, or synthetic weight/input seed",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Pass deterministic=True while loading the checkpoint model",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1.0e-5,
        help="RMSNorm epsilon for --ffn-state or --synthetic mode",
    )
    parser.add_argument(
        "--sr-state",
        help="Optional compact exact FFN persistent-SR state payload",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help=(
            "Explicit SR seed (captured default: NVFP4_RNG_SEED or 0; "
            "synthetic default: 42)"
        ),
    )
    parser.add_argument(
        "--rng-subsequence",
        type=int,
        default=None,
        help=(
            "Explicit SR subsequence base (captured default: "
            "NVFP4_RNG_SUBSEQUENCE_BASE or 0; synthetic default: 17)"
        ),
    )
    parser.add_argument(
        "--gate-seed",
        type=int,
        default=20260828,
        help="CPU/CUDA RNG seed restored before every measured arm",
    )
    parser.add_argument(
        "--scale-num",
        type=float,
        required=True,
        help="Production localCTA scale numerator; this gate requires 448",
    )
    parser.add_argument(
        "--expected-extension",
        required=True,
        help="Absolute path to the freshly built _tk_quant_localcta_v4 module",
    )
    args = parser.parse_args()

    if args.synthetic:
        incompatible = {
            "--io-state": args.io_state,
            "--checkpoint": args.checkpoint,
            "--ffn-state": args.ffn_state,
            "--config": args.config,
            "--layer": args.layer,
            "--rows": args.rows,
            "--sr-state": args.sr_state,
        }
        present = [name for name, value in incompatible.items() if value is not None]
        if present:
            parser.error("--synthetic cannot be combined with " + ", ".join(present))
        if args.deterministic:
            parser.error(
                "--deterministic is a Generator option and invalid with --synthetic"
            )
    else:
        if not args.io_state:
            parser.error("--io-state is required unless --synthetic is selected")
        if not args.checkpoint and not args.ffn_state:
            parser.error(
                "one of --checkpoint or --ffn-state is required unless "
                "--synthetic is selected"
            )
        if args.checkpoint and not args.config:
            parser.error("--config is required with --checkpoint")
        if args.ffn_state and args.config:
            parser.error("--config is only valid with --checkpoint")
    if args.layer is not None and args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.rows is not None and args.rows <= 0:
        parser.error("--rows must be positive")
    if args.norm_eps <= 0:
        parser.error("--norm-eps must be positive")
    return args


def _torch_load_mapping(path: str, label: str) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must contain a mapping, found {type(payload)!r}")
    return payload


def _captured_io(
    path: str,
    *,
    requested_layer: int | None,
    requested_rows: int | None,
) -> tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, int, int]:
    payload = _torch_load_mapping(path, "--io-state")
    missing = {"input", "upstream"} - payload.keys()
    if missing:
        raise ValueError(f"--io-state is missing tensors: {sorted(missing)}")
    input_cpu = payload["input"]
    upstream_cpu = payload["upstream"]
    if not torch.is_tensor(input_cpu) or not torch.is_tensor(upstream_cpu):
        raise TypeError("--io-state input/upstream must be tensors")
    if input_cpu.ndim != 2 or upstream_cpu.ndim != 2:
        raise ValueError(
            "--io-state input/upstream must be 2-D; got "
            f"{tuple(input_cpu.shape)} and {tuple(upstream_cpu.shape)}"
        )
    if tuple(input_cpu.shape) != tuple(upstream_cpu.shape):
        raise ValueError(
            "--io-state input/upstream shapes differ: "
            f"{tuple(input_cpu.shape)} vs {tuple(upstream_cpu.shape)}"
        )
    if not input_cpu.is_floating_point() or not upstream_cpu.is_floating_point():
        raise TypeError("--io-state input/upstream must be floating-point tensors")
    if not bool(torch.isfinite(input_cpu).all()):
        raise ValueError("--io-state input contains non-finite values")
    if not bool(torch.isfinite(upstream_cpu).all()):
        raise ValueError("--io-state upstream contains non-finite values")

    recorded_layer = payload.get("layer")
    if requested_layer is None:
        if not isinstance(recorded_layer, int):
            raise ValueError("--layer is required when --io-state has no integer layer")
        layer = recorded_layer
    else:
        layer = requested_layer
        if recorded_layer is not None and int(recorded_layer) != layer:
            raise ValueError(
                f"requested layer {layer} differs from captured layer {recorded_layer}"
            )

    available_rows, dim = (int(value) for value in input_cpu.shape)
    rows = available_rows if requested_rows is None else requested_rows
    if rows > available_rows:
        raise ValueError(
            f"requested {rows} rows but --io-state contains {available_rows}"
        )
    if rows % 256:
        raise ValueError(
            f"localCTA-v4 consumer gate requires rows divisible by 256, got {rows}"
        )
    return (
        payload,
        input_cpu[:rows].contiguous(),
        upstream_cpu[:rows].contiguous(),
        layer,
        dim,
    )


def _find_ffn(model: torch.nn.Module, layer: int, ffn_type: type):
    suffix = f"layers.{layer}.feed_forward"
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ffn_type)
        and (name == suffix or name.endswith(f".{suffix}"))
    ]
    if len(matches) != 1:
        available = [
            name
            for name, module in model.named_modules()
            if isinstance(module, ffn_type)
        ]
        raise RuntimeError(
            f"expected one converted FFN for layer {layer}, found {matches}; "
            f"available={available}"
        )
    return matches[0]


def _configure_module_parameters(
    module: torch.nn.Module, *, dtype: torch.dtype
) -> None:
    for parameter in module.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
        parameter.requires_grad_(True)
    module.train()


def _load_checkpoint_ffn(
    args: argparse.Namespace,
    *,
    layer: int,
    captured_module_name: object,
):
    from low_bits_training.generate.generate import Generator
    from low_bits_training.quantization.fused_te_linear import (
        FusedFeedForwardFP4_TK,
    )

    generator = Generator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        deterministic=args.deterministic,
        add_kv_cache=False,
        device=args.device,
    )
    model = generator.model
    module_name, module = _find_ffn(model, layer, FusedFeedForwardFP4_TK)
    if captured_module_name is not None and str(captured_module_name) != module_name:
        raise RuntimeError(
            "captured module identity differs from the checkpoint model: "
            f"capture={captured_module_name!r}, loaded={module_name!r}"
        )
    dtype_name = generator.config.training.mixed_precision_param
    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError as error:
        raise ValueError(
            f"unsupported mixed-precision parameter dtype: {dtype_name}"
        ) from error
    if dtype is not torch.bfloat16:
        raise RuntimeError(
            f"production consumer gate requires BF16 parameters, got {dtype_name}"
        )
    _configure_module_parameters(module, dtype=dtype)

    # Child modules do not retain their parents.  Keeping the selected FFN while
    # releasing the rest of the 8B model leaves room for full-production M/H.
    del model, generator
    gc.collect()
    torch.cuda.empty_cache()
    return module_name, module


def _load_compact_ffn(
    args: argparse.Namespace,
    *,
    io_dim: int,
    captured_module_name: object,
):
    from low_bits_training.quantization.fused_te_linear import (
        FusedFeedForwardFP4_TK,
    )

    state = _torch_load_mapping(args.ffn_state, "--ffn-state")
    required = {"norm_weight", "w1_weight", "w3_weight", "w2_weight"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"--ffn-state is missing tensors: {sorted(missing)}")
    for name in required:
        if not torch.is_tensor(state[name]):
            raise TypeError(f"--ffn-state {name} must be a tensor")
    norm = state["norm_weight"]
    w1 = state["w1_weight"]
    w3 = state["w3_weight"]
    w2 = state["w2_weight"]
    if norm.ndim != 1 or int(norm.shape[0]) != io_dim:
        raise ValueError(
            f"norm_weight shape {tuple(norm.shape)} is incompatible with dim={io_dim}"
        )
    if w1.ndim != 2 or w3.ndim != 2 or tuple(w1.shape) != tuple(w3.shape):
        raise ValueError(
            "w1_weight and w3_weight must have the same 2-D shape; got "
            f"{tuple(w1.shape)} and {tuple(w3.shape)}"
        )
    hidden_dim = int(w1.shape[0])
    if tuple(w1.shape) != (hidden_dim, io_dim):
        raise ValueError(
            f"w1/w3 shape {tuple(w1.shape)} is incompatible with dim={io_dim}"
        )
    if tuple(w2.shape) != (io_dim, hidden_dim):
        raise ValueError(f"w2_weight shape {tuple(w2.shape)} != {(io_dim, hidden_dim)}")

    device = torch.device(args.device)
    module = FusedFeedForwardFP4_TK(
        io_dim,
        hidden_dim,
        norm_eps=args.norm_eps,
        device=device,
        dtype=torch.bfloat16,
    )
    targets = {
        "norm_weight": module.norm_weight,
        "w1_weight": module._w1_weight_view(),
        "w3_weight": module._w3_weight_view(),
        "w2_weight": module.w2_weight,
    }
    with torch.no_grad():
        for name, target in targets.items():
            source = state[name]
            if not source.is_floating_point():
                raise TypeError(f"--ffn-state {name} must be floating point")
            target.copy_(source.to(device=device, dtype=torch.bfloat16))
    _configure_module_parameters(module, dtype=torch.bfloat16)
    module_name = (
        str(captured_module_name)
        if isinstance(captured_module_name, str) and captured_module_name
        else "consumer_gate.layers.ffn"
    )
    module._lbt_debug_name = module_name
    return module_name, module


def _make_synthetic_ffn_case(
    args: argparse.Namespace,
    *,
    device: torch.device,
):
    """Build the fixed production-shape diagnostic case on one CUDA device."""
    from low_bits_training.quantization.fused_te_linear import (
        FusedFeedForwardFP4_TK,
    )

    # Match layer 31's normal initialization rule from the Llama FFN module.
    layer_init_std = 0.02 / (2 * (SYNTHETIC_LAYER + 1)) ** 0.5
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        module = FusedFeedForwardFP4_TK(
            SYNTHETIC_DIM,
            SYNTHETIC_HIDDEN_DIM,
            norm_eps=args.norm_eps,
            device=device,
            dtype=torch.bfloat16,
        )
        module.init_weights(layer_init_std)

        data_generator = torch.Generator(device=device)
        data_generator.manual_seed(args.seed + 1)
        input_source = torch.randn(
            (SYNTHETIC_ROWS, SYNTHETIC_DIM),
            device=device,
            dtype=torch.bfloat16,
            generator=data_generator,
        )
        upstream_source = torch.randn(
            (SYNTHETIC_ROWS, SYNTHETIC_DIM),
            device=device,
            dtype=torch.bfloat16,
            generator=data_generator,
        )
        upstream_source.mul_(SYNTHETIC_UPSTREAM_SCALE)

    _configure_module_parameters(module, dtype=torch.bfloat16)
    module_name = f"synthetic.layers.{SYNTHETIC_LAYER}.feed_forward"
    module._lbt_debug_name = module_name
    return module_name, module, input_source, upstream_source


def _assert_production_policy(fte, tk_gemm, *, rows: int) -> None:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(tk_gemm.use_tk_localcta(), "USE_TK_LOCALCTA must be enabled")
    require(
        tk_gemm.get_tk_localcta_variant() == "v4",
        "USE_TK_LOCALCTA_VARIANT must be v4",
    )
    require(
        fte.use_tk_localcta_forward_for_m(rows),
        f"localCTA must be selected for M={rows}",
    )
    require(not fte.use_cuda_graph(), "USE_CUDA_GRAPH must be disabled for this gate")
    require(
        fte.use_tk_localcta_paired_rht_carrier(),
        "USE_TK_LOCALCTA_PAIRED_RHT_CARRIER must be enabled",
    )
    require(
        fte.use_nvfp4_rht_for_role("activation"),
        "activation RHT must be enabled",
    )
    require(fte.use_nvfp4_rht_for_role("grad"), "gradient RHT must be enabled")
    require(
        not fte.use_nvfp4_rht_for_role("weight"),
        "weight RHT must be disabled",
    )
    require(fte._nvfp4_rht_axes() == "col", "NVFP4_RHT_AXES must be col")
    require(
        fte._nvfp4_rht_random_sign_mask(),
        "NVFP4_RHT_RANDOM_SIGNS must be enabled",
    )
    require(
        fte.use_nvfp4_data_stochastic_rounding_for_role("grad"),
        "gradient data SR must be enabled",
    )
    require(fte._nvfp4_grad_sr_axes() == "row", "NVFP4_GRAD_SR_AXES must be row")
    require(
        not fte.use_nvfp4_data_stochastic_rounding_for_role("activation"),
        "activation data SR must be disabled",
    )
    require(
        not fte.use_nvfp4_data_stochastic_rounding_for_role("weight"),
        "weight data SR must be disabled",
    )
    for role in ("activation", "grad", "weight"):
        require(
            not fte.use_nvfp4_scale_stochastic_rounding_for_role(role),
            f"{role} scale SR must be disabled",
        )
    require(
        not fte.use_nvfp4_encode_centric(),
        "NVTE_NVFP4_ENCODE_CENTRIC must be false",
    )
    if failures:
        raise RuntimeError(
            "production localCTA-v4 paired-RHT policy check failed:\n- "
            + "\n- ".join(failures)
        )


def _load_and_validate_extension(args: argparse.Namespace, fte, tk_gemm):
    if args.scale_num != PRODUCTION_SCALE_NUM:
        raise RuntimeError(
            f"consumer promotion gate requires scale numerator "
            f"{PRODUCTION_SCALE_NUM:g}, got {args.scale_num:g}"
        )
    tkq = tk_gemm._get_tk_quant_for_gemm()
    raw_module = getattr(tkq, "_mod", None)
    if raw_module is None:
        raise RuntimeError("localCTA adapter did not expose its loaded extension")
    actual_extension = Path(raw_module.__file__).resolve()
    expected_extension = Path(args.expected_extension).resolve()
    if actual_extension != expected_extension:
        raise RuntimeError(
            f"stale/wrong TK extension loaded: {actual_extension}; "
            f"expected {expected_extension}"
        )
    expected_lbt = (
        Path(__file__).resolve().parents[1]
        / "low_bits_training/quantization/fused_te_linear.py"
    ).resolve()
    actual_lbt = Path(fte.__file__).resolve()
    if actual_lbt != expected_lbt:
        raise RuntimeError(
            f"wrong LBT worktree loaded: {actual_lbt}; expected {expected_lbt}"
        )
    marker = getattr(raw_module, "tk_localcta_silu_deriv_split2_supports_rht", None)
    if marker is None or not marker():
        raise RuntimeError("loaded TK extension lacks native split2 RHT support")
    if not hasattr(tkq, "tk_set_global_scale_num") or not hasattr(
        tkq, "tk_get_global_scale_num"
    ):
        raise RuntimeError("loaded localCTA adapter lacks global-scale control")
    previous_scale_num = float(tkq.tk_get_global_scale_num())
    tkq.tk_set_global_scale_num(args.scale_num)
    applied = float(tkq.tk_get_global_scale_num())
    if applied != args.scale_num:
        raise RuntimeError(
            f"extension applied scale numerator {applied:g}, "
            f"expected {args.scale_num:g}"
        )
    return tkq, raw_module, actual_extension, previous_scale_num


def _logical_sr_keys(debug_name: str) -> tuple[str, str]:
    from low_bits_training.quantization.localcta_sr_state import (
        ffn_deriv_grad_key,
        ffn_w2_grad_key,
    )

    return ffn_w2_grad_key(debug_name), ffn_deriv_grad_key(debug_name)


def _make_sr_state(
    args: argparse.Namespace,
    *,
    debug_name: str,
    device: torch.device,
):
    from low_bits_training.quantization.localcta_sr_state import LocalCTASRState

    logical_keys = _logical_sr_keys(debug_name)
    if args.synthetic:
        # Do not inherit a launch environment's current counter in synthetic
        # mode.  These are explicit, reproducible diagnostic tensors, not a
        # claim about any checkpoint's stochastic stream.
        rng_seed = SYNTHETIC_SR_SEED if args.rng_seed is None else args.rng_seed
        rng_subsequence = (
            SYNTHETIC_SR_SUBSEQUENCE
            if args.rng_subsequence is None
            else args.rng_subsequence
        )
    else:
        rng_seed = (
            int(os.environ.get("NVFP4_RNG_SEED", "0"))
            if args.rng_seed is None
            else args.rng_seed
        )
        rng_subsequence = (
            int(os.environ.get("NVFP4_RNG_SUBSEQUENCE_BASE", "0"))
            if args.rng_subsequence is None
            else args.rng_subsequence
        )
    state = LocalCTASRState(
        logical_keys,
        device=device,
        user_seed=rng_seed,
        user_subsequence_base=rng_subsequence,
        training_steps=1,
        gradient_accumulation_steps=1,
        rank=0,
        world_size=1,
    )
    source = (
        "synthetic_explicit_not_checkpoint_locked"
        if args.synthetic
        else "configured_base"
    )
    if args.sr_state:
        payload = _torch_load_mapping(args.sr_state, "--sr-state")
        missing = set(COMPACT_SR_KEYS) - payload.keys()
        if missing:
            raise ValueError(f"--sr-state is missing tensors: {sorted(missing)}")
        for compact_key, logical_key in zip(COMPACT_SR_KEYS, logical_keys):
            value = payload[compact_key]
            if not torch.is_tensor(value):
                raise TypeError(f"--sr-state {compact_key} must be a tensor")
            if value.dtype != torch.int64 or tuple(value.shape) != (2,):
                raise ValueError(
                    f"--sr-state {compact_key} must be signed int64[2], got "
                    f"dtype={value.dtype}, shape={tuple(value.shape)}"
                )
            state.get(logical_key).copy_(value.to(device=device))
        torch.cuda.synchronize(device)
        source = "checkpoint_compact_file"
    return state, logical_keys, source


def _snapshot_sr_state(state, logical_keys: tuple[str, str]):
    return {key: state.get(key).detach().clone() for key in logical_keys}


def _restore_sr_state(state, snapshots: Mapping[str, torch.Tensor]) -> None:
    for key, snapshot in snapshots.items():
        state.get(key).copy_(snapshot)


def _restore_rng(
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
    device: torch.device,
) -> None:
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device=device)


def _as_uint64(value: int) -> int:
    return int(value) & UINT64_MASK


def _assert_one_sr_reservation(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> None:
    for key in before:
        before_cpu = before[key].detach().cpu()
        after_cpu = after[key].detach().cpu()
        if int(after_cpu[0]) != int(before_cpu[0]):
            raise AssertionError(f"{key}: Philox seed changed during one FFN backward")
        expected = (_as_uint64(int(before_cpu[1])) + SUBSEQUENCE_STRIDE) & UINT64_MASK
        actual = _as_uint64(int(after_cpu[1]))
        if actual != expected:
            raise AssertionError(
                f"{key}: expected one +2^32 reservation; "
                f"before_u64={_as_uint64(int(before_cpu[1]))}, "
                f"after_u64={actual}, expected_u64={expected}"
            )


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype {actual.dtype} != {expected.dtype}")
    actual_bytes = _byte_view(actual)
    expected_bytes = _byte_view(expected)
    if torch.equal(actual_bytes, expected_bytes):
        return
    byte_mismatches = int((actual_bytes != expected_bytes).sum().item())
    element_mismatches = int((actual != expected).sum().item())
    max_abs = None
    if actual.is_floating_point() and expected.is_floating_point():
        max_abs = float(
            (actual.detach().float() - expected.detach().float()).abs().max().item()
        )
    raise AssertionError(
        f"{name}: byte_mismatches={byte_mismatches}/{actual_bytes.numel()}, "
        f"element_mismatches={element_mismatches}/{actual.numel()}, "
        f"max_abs={max_abs}"
    )


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        nonfinite = int((~torch.isfinite(tensor)).sum().item())
        raise AssertionError(f"{name}: found {nonfinite} non-finite elements")


def _grad_norm_diff_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, int | float]:
    """Compute deterministic CPU metrics for the BF16 RMSNorm reduction."""
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            "grad_norm_weight shape mismatch: "
            f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise AssertionError(
            "grad_norm_weight envelope requires BF16 tensors, got "
            f"{actual.dtype} and {expected.dtype}"
        )
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    _assert_finite("grad_norm_weight.actual", actual_cpu)
    _assert_finite("grad_norm_weight.expected", expected_cpu)
    mismatch_count = int((actual_cpu != expected_cpu).sum().item())
    abs_diff = (actual_cpu.float() - expected_cpu.float()).double().abs()
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    mean_abs = float(abs_diff.mean().item()) if abs_diff.numel() else 0.0
    rmse = float(abs_diff.square().mean().sqrt().item()) if abs_diff.numel() else 0.0
    return {
        "mismatched_elements": mismatch_count,
        "total_elements": int(actual_cpu.numel()),
        "mismatch_fraction": (
            mismatch_count / int(actual_cpu.numel()) if actual_cpu.numel() else 0.0
        ),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
    }


def _componentwise_grad_norm_envelope(
    comparisons: list[dict[str, Any]],
) -> dict[str, int | float]:
    if not comparisons:
        raise AssertionError("grad_norm_weight envelope has no comparisons")
    envelope: dict[str, int | float] = {}
    for metric in GRAD_NORM_ENVELOPE_METRICS:
        envelope[metric] = max(comparison[metric] for comparison in comparisons)
    envelope["total_elements"] = comparisons[0]["total_elements"]
    envelope["mismatch_fraction"] = max(
        comparison["mismatch_fraction"] for comparison in comparisons
    )
    return envelope


def _evaluate_grad_norm_weight_envelope(
    arms: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    native = [arm for arm in arms if arm["route"] == "native"]
    fallback = [arm for arm in arms if arm["route"] == "fallback"]
    if len(native) < 2 or not fallback:
        raise AssertionError(
            "grad_norm_weight envelope requires at least two native arms and "
            "one fallback arm"
        )

    native_pairs: list[dict[str, Any]] = []
    for left_position, left in enumerate(native):
        for right in native[left_position + 1 :]:
            metrics = _grad_norm_diff_metrics(right["tensor"], left["tensor"])
            native_pairs.append(
                {
                    "lhs_index": left["index"],
                    "rhs_index": right["index"],
                    **metrics,
                }
            )
    native_envelope = _componentwise_grad_norm_envelope(native_pairs)

    reference = native[0]
    fallback_vs_reference: list[dict[str, Any]] = []
    for arm in fallback:
        metrics = _grad_norm_diff_metrics(arm["tensor"], reference["tensor"])
        fallback_vs_reference.append(
            {
                "reference_native_index": reference["index"],
                "fallback_index": arm["index"],
                **metrics,
            }
        )
    fallback_envelope = _componentwise_grad_norm_envelope(fallback_vs_reference)

    failures: list[str] = []
    if float(native_envelope["max_abs"]) > GRAD_NORM_WEIGHT_ABS_CAP:
        failures.append(
            "native/native grad_norm_weight max_abs exceeds the BF16 absolute "
            f"cap: {native_envelope['max_abs']} > {GRAD_NORM_WEIGHT_ABS_CAP}"
        )
    for comparison in fallback_vs_reference:
        if float(comparison["max_abs"]) > GRAD_NORM_WEIGHT_ABS_CAP:
            failures.append(
                "fallback/reference grad_norm_weight max_abs exceeds the BF16 "
                f"absolute cap for arm {comparison['fallback_index']}: "
                f"{comparison['max_abs']} > {GRAD_NORM_WEIGHT_ABS_CAP}"
            )
        for metric in GRAD_NORM_ENVELOPE_METRICS:
            if comparison[metric] > native_envelope[metric]:
                failures.append(
                    "fallback/reference grad_norm_weight exceeds measured "
                    f"native/native envelope for arm "
                    f"{comparison['fallback_index']} metric={metric}: "
                    f"{comparison[metric]} > {native_envelope[metric]}"
                )

    report = {
        "policy": (
            "all fallback/reference metrics must be <= the component-wise "
            "all-pairs native/native envelope; native and fallback max_abs "
            "must also be <= the fixed BF16 cap"
        ),
        "absolute_max_cap": GRAD_NORM_WEIGHT_ABS_CAP,
        "envelope_metrics": list(GRAD_NORM_ENVELOPE_METRICS),
        "native_indices": [arm["index"] for arm in native],
        "fallback_indices": [arm["index"] for arm in fallback],
        "native_vs_native_pairs": native_pairs,
        "native_vs_native_envelope": native_envelope,
        "a_a_envelope": native_envelope,
        "fallback_vs_reference": fallback_vs_reference,
        "fallback_vs_reference_envelope": fallback_envelope,
        "a_b_envelope": fallback_envelope,
        "passed": not failures,
        "failures": failures,
    }
    return report, failures


def _set_route_and_assert(fte, raw_module, route: str) -> None:
    os.environ[SELECTOR_ENV] = "1" if route == "native" else "0"
    selected = fte._use_tk_localcta_native_paired_rht_split2(
        raw_module,
        paired_rht_carrier=fte.use_tk_localcta_paired_rht_carrier(),
    )
    expected = route == "native"
    if selected != expected:
        raise RuntimeError(
            f"route selector requested {route}, but production selector returned "
            f"native={selected}"
        )


def _run_arm(
    *,
    route: str,
    module,
    input_source: torch.Tensor,
    upstream_source: torch.Tensor,
    sr_state,
    logical_keys: tuple[str, str],
    initial_sr: Mapping[str, torch.Tensor],
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor,
    device: torch.device,
    fte,
    raw_module,
    capture: bool,
):
    _restore_sr_state(sr_state, initial_sr)
    _restore_rng(cpu_rng_state, cuda_rng_state, device)
    torch.cuda.synchronize(device)
    _set_route_and_assert(fte, raw_module, route)

    x = input_source.detach().clone().requires_grad_(True)
    upstream = upstream_source.detach().clone()
    output = module.forward_with_residual(x, residual=x)
    primary = output[0] if isinstance(output, tuple) else output
    if tuple(primary.shape) != tuple(upstream.shape):
        raise RuntimeError(
            f"FFN output shape {tuple(primary.shape)} != upstream "
            f"shape {tuple(upstream.shape)}"
        )
    grad_inputs = (
        x,
        module.norm_weight,
        module._w1_weight_view(),
        module._w3_weight_view(),
        module.w2_weight,
    )
    grads = torch.autograd.grad(primary, grad_inputs, grad_outputs=upstream)
    torch.cuda.synchronize(device)
    after_sr = _snapshot_sr_state(sr_state, logical_keys)
    _assert_one_sr_reservation(initial_sr, after_sr)
    _assert_exact(f"{route}.input_unchanged", x.detach(), input_source)
    _assert_exact(f"{route}.upstream_unchanged", upstream, upstream_source)

    values = (primary, *grads)
    for name, value in zip(RESULT_NAMES, values):
        _assert_finite(f"{route}.{name}", value)
    pointers = {
        name: int(value.data_ptr()) for name, value in zip(RESULT_NAMES, values)
    }
    snapshots = (
        {name: value.detach().clone() for name, value in zip(RESULT_NAMES, values)}
        if capture
        else None
    )
    return snapshots, after_sr, pointers


def _state_json(state: Mapping[str, torch.Tensor], logical_keys: tuple[str, str]):
    return {
        compact: [int(value) for value in state[logical].detach().cpu().tolist()]
        for compact, logical in zip(COMPACT_SR_KEYS, logical_keys)
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"consumer equivalence gate requires CUDA, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        torch.cuda.set_device(device)

    if args.scale_num != PRODUCTION_SCALE_NUM:
        raise RuntimeError(
            f"consumer promotion gate requires scale numerator "
            f"{PRODUCTION_SCALE_NUM:g}, got {args.scale_num:g}"
        )
    os.environ["USE_TK_LOCALCTA_SCALE_NUM"] = f"{args.scale_num:g}"

    if args.synthetic:
        print(
            "SYNTHETIC_DIAGNOSTIC_ONLY checkpoint_lock_gate="
            "FAIL_SYNTHETIC_NOT_CHECKPOINT_LOCKED; this result cannot promote "
            "a checkpoint continuation",
            file=sys.stderr,
            flush=True,
        )
        io_payload: Mapping[str, Any] = {}
        input_cpu = upstream_cpu = None
        layer = SYNTHETIC_LAYER
        dim = SYNTHETIC_DIM
        case_rows = SYNTHETIC_ROWS
    else:
        io_payload, input_cpu, upstream_cpu, layer, dim = _captured_io(
            args.io_state,
            requested_layer=args.layer,
            requested_rows=args.rows,
        )
        case_rows = int(input_cpu.shape[0])

    # Import route-sensitive production modules only after argument validation
    # and scale configuration.  All other recipe settings come from the caller.
    from low_bits_training.quantization import fused_te_linear as fte
    from low_bits_training.quantization import tk_gemm
    from low_bits_training.quantization.localcta_sr_state import (
        active_localcta_sr_state,
        set_active_localcta_sr_state,
    )

    _assert_production_policy(fte, tk_gemm, rows=case_rows)
    tkq, raw_module, extension_path, previous_scale_num = _load_and_validate_extension(
        args, fte, tk_gemm
    )

    previous_selector = os.environ.get(SELECTOR_ENV)
    previous_active_sr = active_localcta_sr_state()
    try:
        if args.synthetic:
            module_name, module, input_source, upstream_source = (
                _make_synthetic_ffn_case(args, device=device)
            )
        elif args.checkpoint:
            module_name, module = _load_checkpoint_ffn(
                args,
                layer=layer,
                captured_module_name=io_payload.get("module_name"),
            )
        else:
            module_name, module = _load_compact_ffn(
                args,
                io_dim=dim,
                captured_module_name=io_payload.get("module_name"),
            )
        if module.dim != dim:
            raise RuntimeError(
                f"captured FFN dim {dim} differs from loaded module dim {module.dim}"
            )
        debug_name = getattr(module, "_lbt_debug_name", None)
        if not isinstance(debug_name, str) or not debug_name:
            debug_name = module_name
            module._lbt_debug_name = debug_name

        if not args.synthetic:
            assert input_cpu is not None and upstream_cpu is not None
            input_source = input_cpu.to(
                device=device, dtype=torch.bfloat16, non_blocking=False
            ).contiguous()
            upstream_source = upstream_cpu.to(
                device=device, dtype=torch.bfloat16, non_blocking=False
            ).contiguous()
        sr_state, logical_keys, sr_source = _make_sr_state(
            args,
            debug_name=debug_name,
            device=device,
        )
        set_active_localcta_sr_state(sr_state)

        torch.manual_seed(args.gate_seed)
        torch.cuda.manual_seed_all(args.gate_seed)
        cpu_rng_state = torch.get_rng_state().clone()
        cuda_rng_state = torch.cuda.get_rng_state(device).clone()
        initial_sr = _snapshot_sr_state(sr_state, logical_keys)

        # Start this standalone gate from empty LBT step caches, then leave the
        # caches alive across both warmups and every measured arm.
        fte.clear_fused_fp4_step_caches()
        for route in ("native", "fallback"):
            _run_arm(
                route=route,
                module=module,
                input_source=input_source,
                upstream_source=upstream_source,
                sr_state=sr_state,
                logical_keys=logical_keys,
                initial_sr=initial_sr,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
                device=device,
                fte=fte,
                raw_module=raw_module,
                capture=False,
            )
        _restore_sr_state(sr_state, initial_sr)
        _restore_rng(cpu_rng_state, cuda_rng_state, device)
        torch.cuda.synchronize(device)

        reference: Mapping[str, torch.Tensor] | None = None
        reference_state: Mapping[str, torch.Tensor] | None = None
        arms: list[dict[str, Any]] = []
        grad_norm_weight_arms: list[dict[str, Any]] = []
        for index, route in enumerate(MEASURED_ROUTES):
            snapshots, after_sr, pointers = _run_arm(
                route=route,
                module=module,
                input_source=input_source,
                upstream_source=upstream_source,
                sr_state=sr_state,
                logical_keys=logical_keys,
                initial_sr=initial_sr,
                cpu_rng_state=cpu_rng_state,
                cuda_rng_state=cuda_rng_state,
                device=device,
                fte=fte,
                raw_module=raw_module,
                capture=True,
            )
            assert snapshots is not None
            grad_norm_weight_arms.append(
                {
                    "index": index,
                    "route": route,
                    "tensor": snapshots["grad_norm_weight"].detach().cpu().clone(),
                }
            )
            if reference is None:
                reference = {name: snapshots[name] for name in EXACT_RESULT_NAMES}
                reference_state = after_sr
            else:
                for name in EXACT_RESULT_NAMES:
                    comparison = (
                        f"native_repeat[{index}].{name}"
                        if route == "native"
                        else f"native_vs_fallback[{index}].{name}"
                    )
                    _assert_exact(comparison, snapshots[name], reference[name])
                assert reference_state is not None
                for key in logical_keys:
                    comparison = (
                        f"native_repeat[{index}].sr.{key}"
                        if route == "native"
                        else f"native_vs_fallback[{index}].sr.{key}"
                    )
                    _assert_exact(comparison, after_sr[key], reference_state[key])
            arms.append({"index": index, "route": route, "pointers": pointers})
            del snapshots

        assert reference is not None and reference_state is not None
        grad_norm_weight_report, grad_norm_weight_failures = (
            _evaluate_grad_norm_weight_envelope(grad_norm_weight_arms)
        )
        if grad_norm_weight_failures:
            print(
                json.dumps(
                    {
                        "status": "FAIL_GRAD_NORM_WEIGHT_ENVELOPE",
                        "checkpoint_locked": (
                            False if args.synthetic else bool(args.sr_state)
                        ),
                        "grad_norm_weight": grad_norm_weight_report,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise AssertionError(
                "grad_norm_weight fallback/reference exceeded the measured "
                "native/native BF16 reduction envelope"
            )
        if args.synthetic:
            status = "PASS_SYNTHETIC_EQUIVALENCE_NOT_CHECKPOINT_LOCKED"
            gate_name = "synthetic_ffn_native_fallback_consumer_equivalence"
            checkpoint_locked = False
            checkpoint_lock_gate = "FAIL_SYNTHETIC_NOT_CHECKPOINT_LOCKED"
            input_source_kind = "synthetic_fixed_production_shape"
            synthetic_spec = {
                "rows_m": SYNTHETIC_ROWS,
                "model_dim_k": SYNTHETIC_DIM,
                "ffn_hidden_h": SYNTHETIC_HIDDEN_DIM,
                "layer": SYNTHETIC_LAYER,
                "tensor_seed": args.seed,
                "layer_init_std": 0.02 / (2 * (SYNTHETIC_LAYER + 1)) ** 0.5,
                "upstream_scale": SYNTHETIC_UPSTREAM_SCALE,
                "norm_eps": args.norm_eps,
                "weight_dtype": "torch.bfloat16",
                "input_dtype": "torch.bfloat16",
            }
        else:
            status = "PASS"
            gate_name = "real_ffn_native_fallback_consumer_equivalence"
            checkpoint_locked = bool(args.sr_state)
            checkpoint_lock_gate = (
                "PASS_CAPTURED_IO_WEIGHTS_AND_SR_STATE"
                if args.sr_state
                else "NOT_REQUESTED_CONFIGURED_BASE_SR_STATE"
            )
            input_source_kind = "checkpoint_capture"
            synthetic_spec = None
        print(
            json.dumps(
                {
                    "status": status,
                    "gate": gate_name,
                    "checkpoint_locked": checkpoint_locked,
                    "checkpoint_lock_gate": checkpoint_lock_gate,
                    "input_source_kind": input_source_kind,
                    "synthetic_spec": synthetic_spec,
                    "extension": str(extension_path),
                    "module_name": module_name,
                    "debug_name": debug_name,
                    "layer": layer,
                    "rows": int(input_source.shape[0]),
                    "dim": int(module.dim),
                    "hidden_dim": int(module.hidden_dim),
                    "dtype": str(input_source.dtype),
                    "scale_num": args.scale_num,
                    "gate_seed": args.gate_seed,
                    "sr_source": sr_source,
                    "initial_sr_state": _state_json(initial_sr, logical_keys),
                    "advanced_sr_state": _state_json(reference_state, logical_keys),
                    "grad_norm_weight": grad_norm_weight_report,
                    "measured_order": list(MEASURED_ROUTES),
                    "arms": arms,
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
