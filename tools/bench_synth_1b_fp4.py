#!/usr/bin/env python3
"""Synthetic end-to-end 1B FP4 step benchmark.

Runs one forward+backward training step on a synthetic token batch using the
actual FP4-converted Llama model, without requiring a dataset/tokenizer.

Examples:
  CUDA_VISIBLE_DEVICES=1 python tools/bench_synth_1b_fp4.py \
      --mode fp4_localcta_fused --flavor 1B --batch-size 64 --seq-len 1024

  CUDA_VISIBLE_DEVICES=1 python tools/bench_synth_1b_fp4.py \
      --sweep --flavor 1B_legacy --batch-sizes 4 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace


REPO_ROOT = os.path.abspath(
    os.environ.get(
        "LOW_BITS_TRAINING_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
)
TORCHTITAN_ROOT = os.path.join(REPO_ROOT, "torchtitan_submodule")
FALLBACK_TORCHTITAN_ROOT = "/opt/mfu/EXTERNAL_PATH"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, TORCHTITAN_ROOT)
if os.path.isdir(FALLBACK_TORCHTITAN_ROOT) and FALLBACK_TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, FALLBACK_TORCHTITAN_ROOT)


MODES = [
    "bf16",
    "fp4_fused_te",
    "fp4_tk",
    "fp4_localcta",
    "fp4_localcta_fused",
    "te_nvfp4_native",
    "te_mxfp4_native",
    "mxfp4_tk_backend",
    "mxfp4_tk_fused",
]

REFERENCE_PAIRS = {
    "qkv_fast": {
        "label": "qkv_fast_5eea091_21ed5e27",
        "fp4_commit": "5eea091",
        "low_bits_commit": "21ed5e27",
        "low_bits_roots": [
            "/tmp/lbt_ws_21ed5e27",
        ],
        "fp4_roots": [
            "/tmp/fp4_ws_5eea091",
        ],
    },
    "ffn_fast": {
        "label": "ffn_fast_0048f25_b1d2bfd2_or_f555f454",
        "fp4_commit": "0048f25",
        "low_bits_commit": "b1d2bfd2/f555f454",
        "low_bits_roots": [
            "/tmp/lbt_ws_f555f454",
            "/tmp/lbt_ws_b1d2bfd2",
        ],
        "fp4_roots": [
            "/tmp/fp4_ws_0048f25",
        ],
    },
    "current_with_qkv_fast_parent": {
        "label": "current_low_bits_with_qkv_fast_parent",
        "fp4_commit": "5eea091",
        "low_bits_commit": "current",
        "low_bits_roots": [
            REPO_ROOT,
        ],
        "fp4_roots": [
            "/tmp/fp4_ws_5eea091",
        ],
    },
    "current_with_ffn_fast_parent": {
        "label": "current_low_bits_with_ffn_fast_parent",
        "fp4_commit": "0048f25",
        "low_bits_commit": "current",
        "low_bits_roots": [
            REPO_ROOT,
        ],
        "fp4_roots": [
            "/tmp/fp4_ws_0048f25",
        ],
    },
    "current_with_ws_sync_parent": {
        "label": "current_low_bits_with_ws_sync_parent",
        "fp4_commit": "ws-sync-main-0330",
        "low_bits_commit": "current",
        "low_bits_roots": [
            REPO_ROOT,
        ],
        "fp4_roots": [
            "/opt/mfu/EXTERNAL_PATH",
        ],
    },
}


def _first_existing_path(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def _installed_te_root() -> str | None:
    try:
        import transformer_engine  # type: ignore
    except Exception:
        return None
    return os.path.abspath(transformer_engine.__path__[0])


def _ensure_historical_te_shim(low_bits_root: str) -> None:
    te_root = _installed_te_root()
    if te_root is None:
        return
    te_lib = os.path.join(te_root, "libtransformer_engine.so")
    if not os.path.exists(te_lib):
        return
    shim_root = os.path.join(low_bits_root, "TransformerEngine")
    shim_lib_dir = os.path.join(shim_root, "build", "cmake")
    shim_pkg = os.path.join(shim_root, "transformer_engine")
    os.makedirs(shim_lib_dir, exist_ok=True)
    shim_lib = os.path.join(shim_lib_dir, "libtransformer_engine.so")
    if not os.path.exists(shim_lib):
        os.symlink(te_lib, shim_lib)
    if not os.path.exists(shim_pkg):
        os.symlink(te_root, shim_pkg)


def _has_localcta_runtime_artifacts(fp4_root: str) -> bool:
    quant_so = os.path.join(
        fp4_root,
        "TK_quantisation",
        "nvfp4_CTA_local_v1",
        "_tk_quant_localcta.cpython-312-aarch64-linux-gnu.so",
    )
    gemm_so = os.path.join(
        fp4_root,
        "ThunderKittens",
        "kernels",
        "gemm",
        "nvfp4_b200",
        "localCTA_epilogue",
        "_C_nv_localcta_gemm.cpython-312-aarch64-linux-gnu.so",
    )
    return os.path.exists(quant_so) and os.path.exists(gemm_so)


def _set_tmp_fp4_matmul_symlink(fp4_root: str) -> tuple[bool, str | None]:
    link_path = "/tmp/fp4_matmul"
    previous = None
    if os.path.lexists(link_path):
        if os.path.islink(link_path):
            previous = os.readlink(link_path)
        else:
            raise RuntimeError(f"{link_path} exists and is not a symlink")
        os.unlink(link_path)
    os.symlink(fp4_root, link_path)
    return True, previous


def _restore_tmp_fp4_matmul_symlink(previous: str | None) -> None:
    link_path = "/tmp/fp4_matmul"
    if os.path.lexists(link_path):
        os.unlink(link_path)
    if previous is not None:
        os.symlink(previous, link_path)


def configure_env(mode: str, mxfp4_backend_version: str = "v4") -> None:
    os.environ.setdefault("NVTE_NVFP4_DISABLE_RHT", "1")
    os.environ.setdefault("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "1")
    os.environ.setdefault("NVTE_NVFP4_ENCODE_CENTRIC", "0")
    os.environ.setdefault("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "1")
    os.environ.setdefault("NVTE_CUSTOM_QUANT", "0")
    os.environ.setdefault("FUSED_TE_QUANT", "0")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    env = {
        "USE_TK_GEMM": "0",
        "USE_TK_QUANT": "0",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_FUSED": "0",
        "USE_TK_LOCALCTA_FFN_EXPERIMENT": "off",
        "USE_TK_LOCALCTA_FFN_FUSED_ROW_PRODUCER": "0",
        "USE_MXFP4_TK_BACKEND": "0",
        "USE_MXFP4_TK_FUSED": "0",
        "MXFP4_BACKEND_VERSION": mxfp4_backend_version,
        "MXFP4_USE_FUSED_RMSNORM_QUANT": os.environ.get("MXFP4_USE_FUSED_RMSNORM_QUANT", "0"),
        "MXFP4_USE_FUSED_RMSNORM_QUANT_QKV": os.environ.get("MXFP4_USE_FUSED_RMSNORM_QUANT_QKV", "1"),
        "MXFP4_USE_FUSED_RMSNORM_QUANT_FFN": os.environ.get("MXFP4_USE_FUSED_RMSNORM_QUANT_FFN", "1"),
        "MXFP4_USE_SPLIT3_QKV_QUANT": os.environ.get("MXFP4_USE_SPLIT3_QKV_QUANT", "1"),
        "MXFP4_USE_QKV_DIRECT_OUTPUTS": os.environ.get("MXFP4_USE_QKV_DIRECT_OUTPUTS", "1"),
        "MXFP4_USE_SPLIT3_QKV_STAGE_COPY": os.environ.get("MXFP4_USE_SPLIT3_QKV_STAGE_COPY", "0"),
        "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": os.environ.get("MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD", "0"),
        "MXFP4_QKV_BWD_STATE_SLOTS": os.environ.get("MXFP4_QKV_BWD_STATE_SLOTS", "4"),
        "MXFP4_USE_SPLIT2_FFN_QUANT": os.environ.get("MXFP4_USE_SPLIT2_FFN_QUANT", "1"),
        "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN": os.environ.get("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN", "0"),
        "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": os.environ.get("MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD", "1"),
    }
    if mode == "bf16":
        pass
    elif mode == "fp4_fused_te":
        env["FUSED_TE_QUANT"] = "1"
    elif mode == "fp4_tk":
        env["USE_TK_GEMM"] = "1"
        env["USE_TK_QUANT"] = "1"
    elif mode == "fp4_localcta":
        env["USE_TK_GEMM"] = "1"
        env["USE_TK_QUANT"] = "1"
        env["USE_TK_LOCALCTA"] = "1"
    elif mode == "fp4_localcta_fused":
        env["USE_TK_GEMM"] = "1"
        env["USE_TK_QUANT"] = "1"
        env["USE_TK_LOCALCTA"] = "1"
        env["USE_TK_LOCALCTA_FUSED"] = "1"
    elif mode in ("te_nvfp4_native", "te_mxfp4_native"):
        pass
    elif mode == "mxfp4_tk_backend":
        env["USE_MXFP4_TK_BACKEND"] = "1"
    elif mode == "mxfp4_tk_fused":
        env["USE_MXFP4_TK_FUSED"] = "1"
    else:
        raise ValueError(f"unknown mode: {mode}")

    for k, v in env.items():
        os.environ[k] = v


def _preinit_cuda(device_index: int = 0) -> None:
    import ctypes
    import torch

    cudart = None
    for path in (
        "/usr/local/cuda/lib64/libcudart.so",
        "libcudart.so",
    ):
        try:
            cudart = ctypes.CDLL(path)
            break
        except OSError:
            continue

    last_exc = None
    for _ in range(5):
        try:
            if cudart is not None:
                cudart.cudaSetDevice.argtypes = [ctypes.c_int]
                cudart.cudaSetDevice.restype = ctypes.c_int
                cudart.cudaFree.argtypes = [ctypes.c_void_p]
                cudart.cudaFree.restype = ctypes.c_int
                cudart.cudaSetDevice(device_index)
                cudart.cudaFree(None)
            torch.empty(1, device=f"cuda:{device_index}")
            torch.cuda.synchronize()
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    if last_exc is not None:
        msg = str(last_exc)
        if (
            os.environ.get("BENCH_SYNTH_ALLOW_PREINIT_FAILURE", "0") == "1"
            and "Error 304" in msg
        ):
            print(
                json.dumps(
                    {
                        "bench_warning": "cuda_preinit_failed_but_continuing",
                        "device_index": device_index,
                        "error": msg,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            return
        raise last_exc


def _clear_cuda_runtime_state(device_index: int = 0) -> None:
    import ctypes

    cudart = None
    for path in (
        "/usr/local/cuda/lib64/libcudart.so",
        "libcudart.so",
    ):
        try:
            cudart = ctypes.CDLL(path)
            break
        except OSError:
            continue
    if cudart is None:
        return

    cudart.cudaGetLastError.argtypes = []
    cudart.cudaGetLastError.restype = ctypes.c_int
    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    try:
        cudart.cudaGetLastError()
        cudart.cudaSetDevice(device_index)
        cudart.cudaFree(None)
        cudart.cudaGetLastError()
    except Exception:
        return


def _set_device_with_retry(device_index: int) -> None:
    import torch

    last_exc = None
    for _ in range(5):
        try:
            torch.cuda.set_device(device_index)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    if last_exc is not None:
        raise last_exc


def _has_single_visible_cuda_device() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return False
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    return len(devices) == 1


def _should_skip_explicit_device_init(device_index: int) -> bool:
    if device_index != 0:
        return False
    if not _has_single_visible_cuda_device():
        return False
    return (
        os.environ.get("TK_SKIP_CUDA_PREFLIGHT", "0") == "1"
        or os.environ.get("BENCH_SYNTH_SKIP_EXPLICIT_DEVICE_INIT", "0") == "1"
    )


def _runtime_backend_info(mode: str) -> dict:
    info = {
        "fp4_matmul_root": os.path.abspath(
            os.environ.get("FP4_MATMUL_ROOT", os.path.join(REPO_ROOT, "..", "fp4_matmul"))
        )
    }
    if mode in {"fp4_tk", "fp4_localcta", "fp4_localcta_fused"}:
        from low_bits_training.quantization import tk_gemm

        info["tk"] = tk_gemm.get_tk_backend_info()
    return info


def _model_args_for_flavor(flavor: str):
    from low_bits_training.models.llama3 import TransformerModelArgs

    if flavor == "8B":
        return TransformerModelArgs(
            dim=4096,
            n_layers=32,
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            rope_scaling_args=None,
            max_seq_len=8192,
        )
    if flavor == "1B_legacy":
        return TransformerModelArgs(
            dim=2048,
            n_layers=24,
            n_heads=32,
            n_kv_heads=32,
            ffn_dim_multiplier=1.0,
            multiple_of=256,
            rope_theta=10000,
            vocab_size=32000,
        )
    if flavor == "1B":
        return TransformerModelArgs(
            dim=2048,
            n_layers=16,
            n_heads=32,
            n_kv_heads=8,
            ffn_dim_multiplier=8192 / 4 / 2048 * 3 / 2,
            multiple_of=256,
            rope_theta=500000,
            vocab_size=32000,
        )
    raise ValueError(f"unknown flavor: {flavor}")


def _te_job_config(flavor: str, recipe_name: str):
    n_layers = {"1B_legacy": 24, "1B": 16, "8B": 32}[flavor]
    return SimpleNamespace(
        model=SimpleNamespace(n_layers=n_layers, flavor=flavor),
        te_fp4=SimpleNamespace(
            mlp_recipe=recipe_name,
            attn_recipe=recipe_name,
            exclude_last_n_layers=0,
        ),
    )


def build_model(flavor: str, mode: str, device: str, device_index: int = 0):
    import torch
    import low_bits_training  # noqa: F401
    from low_bits_training.converters import Bfloat16Converter
    from low_bits_training.models.llama3 import Transformer
    from low_bits_training.quantization.float32_linear import Float32Linear

    model_args = _model_args_for_flavor(flavor)
    if device.startswith("cuda"):
        _clear_cuda_runtime_state(device_index)

    model = Transformer(model_args).to(device=device)
    Bfloat16Converter.llama3_gc(model)

    if mode in ("fp4_fused_te", "fp4_tk", "fp4_localcta", "fp4_localcta_fused"):
        from low_bits_training.quantization.fp4_converter import FP4Converter

        class _DummyCfg:
            pass

        FP4Converter(_DummyCfg(), None).convert(model)
    elif mode == "te_nvfp4_native":
        from low_bits_training.quantization.mxfp_custom_te_fp4 import TEFP4Converter

        TEFP4Converter(_te_job_config(flavor, "NVFP4"), None).convert(model)
    elif mode == "te_mxfp4_native":
        from low_bits_training.quantization.mxfp_custom_te_fp4 import TEFP4Converter

        TEFP4Converter(_te_job_config(flavor, "MXFP4"), None).convert(model)
    elif mode == "mxfp4_tk_backend":
        from low_bits_training.quantization.mxfp4_tk_converter import MXFP4TKBackendConverter

        MXFP4TKBackendConverter(None, None).convert(model)
    elif mode == "mxfp4_tk_fused":
        from low_bits_training.quantization.mxfp4_tk_converter import MXFP4TKFusedConverter

        MXFP4TKFusedConverter(None, None).convert(model)

    if mode != "bf16":
        for module in model.modules():
            if isinstance(module, Float32Linear):
                module.to(device=device, dtype=torch.float32)

    return model, model_args


def _get_layer0_modules(model):
    block = model.layers["0"]
    attn = block.attention.fused if hasattr(block.attention, "fused") else block.attention
    ffn = block.feed_forward
    return block, attn, ffn


def get_shape_metadata(model, model_args):
    _, attn, ffn = _get_layer0_modules(model)
    return {
        "dim": model_args.dim,
        "q_dim": attn.q_dim,
        "k_dim": attn.k_dim,
        "v_dim": attn.v_dim,
        "hidden_dim": getattr(ffn, "hidden_dim", None),
    }


def build_isolation_step(model, model_args, block_kind: str, isolation_m: int, device: str):
    import torch

    _, attn, ffn = _get_layer0_modules(model)
    shape = get_shape_metadata(model, model_args)
    meta = {
        "block": block_kind,
        "isolation_m": isolation_m,
        **shape,
    }

    if block_kind == "qkv":
        x_qkv = torch.randn(
            isolation_m, model_args.dim,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )
        gq = torch.randn(isolation_m, attn.q_dim, device=device, dtype=torch.bfloat16)
        gk = torch.randn(isolation_m, attn.k_dim, device=device, dtype=torch.bfloat16)
        gv = torch.randn(isolation_m, attn.v_dim, device=device, dtype=torch.bfloat16)

        def step():
            attn.zero_grad(set_to_none=True)
            if x_qkv.grad is not None:
                x_qkv.grad = None
            q, k, v = attn.forward_qkv(x_qkv)
            torch.autograd.backward((q, k, v), (gq, gk, gv))

        return step, meta

    if block_kind == "wo":
        x_wo = torch.randn(
            isolation_m, attn.q_dim,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )
        gout_wo = torch.randn(isolation_m, model_args.dim, device=device, dtype=torch.bfloat16)

        def step():
            attn.zero_grad(set_to_none=True)
            if x_wo.grad is not None:
                x_wo.grad = None
            y = attn.forward_wo(x_wo)
            torch.autograd.backward(y, gout_wo)

        return step, meta

    if block_kind == "ffn":
        x_ffn = torch.randn(
            isolation_m, model_args.dim,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )
        gout_ffn = torch.randn(isolation_m, model_args.dim, device=device, dtype=torch.bfloat16)

        def step():
            ffn.zero_grad(set_to_none=True)
            if x_ffn.grad is not None:
                x_ffn.grad = None
            y = ffn(x_ffn)
            torch.autograd.backward(y, gout_ffn)

        return step, meta

    raise ValueError(f"unknown block_kind: {block_kind}")


def benchmark_block_step(step, warmup: int, steps: int):
    import torch

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    times = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {"total_ms": times[len(times) // 2]}


def benchmark(model, model_args, batch_size: int, seq_len: int, warmup: int, steps: int):
    import torch

    device = next(model.parameters()).device
    vocab_size = model_args.vocab_size

    torch.manual_seed(1234)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    def run_step():
        model.zero_grad(set_to_none=True)
        start_f = torch.cuda.Event(enable_timing=True)
        end_f = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)

        start_f.record()
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        end_f.record()
        start_b.record()
        loss.backward()
        end_b.record()
        return start_f, end_f, start_b, end_b, float(loss.detach().item())

    torch.cuda.synchronize()
    for _ in range(warmup):
        run_step()
        torch.cuda.synchronize()

    times = []
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        sf, ef, sb, eb, loss_val = run_step()
        torch.cuda.synchronize()
        f_ms = sf.elapsed_time(ef)
        b_ms = sb.elapsed_time(eb)
        times.append((f_ms, b_ms, f_ms + b_ms))
        losses.append(loss_val)

    times.sort(key=lambda x: x[2])
    median = times[len(times) // 2]
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    toks_per_s = (batch_size * seq_len) / (median[2] / 1e3)
    return {
        "forward_ms": median[0],
        "backward_ms": median[1],
        "total_ms": median[2],
        "tokens_per_s": toks_per_s,
        "peak_mem_gib": peak_gib,
        "loss_median": sorted(losses)[len(losses) // 2],
    }


def run_one(
    mode: str,
    flavor: str,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    device_index: int = 0,
    mxfp4_backend_version: str = "v4",
    block: str = "full",
    isolation_m: int = 65536,
):
    import torch

    configure_env(mode, mxfp4_backend_version)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if _should_skip_explicit_device_init(device_index):
        device = "cuda"
    else:
        _preinit_cuda(device_index)
        _set_device_with_retry(device_index)
        device = f"cuda:{device_index}"

    t0 = time.time()
    model, model_args = build_model(flavor, mode, device, device_index=device_index)
    build_s = time.time() - t0
    shape = get_shape_metadata(model, model_args)
    if block == "full":
        stats = benchmark(model, model_args, batch_size, seq_len, warmup, steps)
    else:
        step_fn, block_meta = build_isolation_step(model, model_args, block, isolation_m, device)
        stats = benchmark_block_step(step_fn, warmup, steps)
        stats.update(block_meta)
    return {
        "mode": mode,
        "flavor": flavor,
        "block": block,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "device_index": device_index,
        "mxfp4_backend_version": mxfp4_backend_version,
        "isolation_m": isolation_m if block != "full" else None,
        "build_s": build_s,
        "backend_info": _runtime_backend_info(mode),
        **shape,
        **stats,
    }


def _extract_last_json_blob(output: str):
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            return json.loads(line)
    raise RuntimeError(f"No JSON blob found in output:\n{output}")


def run_reference_compare(
    mode: str,
    flavor: str,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    device_index: int,
    mxfp4_backend_version: str,
    block: str,
    isolation_m: int,
    reference_tags: list[str],
):
    runs = {
        "current": {
            "label": "current",
            "low_bits_root": REPO_ROOT,
            "fp4_matmul_root": os.environ.get(
                "FP4_MATMUL_ROOT",
                os.path.abspath(os.path.join(REPO_ROOT, "..", "fp4_matmul")),
            ),
        }
    }
    for tag in reference_tags:
        spec = REFERENCE_PAIRS[tag]
        low_bits_root = _first_existing_path(spec["low_bits_roots"])
        fp4_root = _first_existing_path(spec["fp4_roots"])
        runs[tag] = {
            "label": spec["label"],
            "low_bits_commit": spec["low_bits_commit"],
            "fp4_commit": spec["fp4_commit"],
            "low_bits_root": low_bits_root,
            "fp4_matmul_root": fp4_root,
        }

    tool_path = os.path.abspath(__file__)
    base_cmd = [
        sys.executable,
        "-u",
        tool_path,
        "--mode",
        mode,
        "--flavor",
        flavor,
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
        "--device-index",
        str(device_index),
        "--mxfp4-backend-version",
        mxfp4_backend_version,
        "--block",
        block,
        "--isolation-m",
        str(isolation_m),
    ]

    results = {
        "mode": mode,
        "flavor": flavor,
        "block": block,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "device_index": device_index,
        "mxfp4_backend_version": mxfp4_backend_version,
        "isolation_m": isolation_m if block != "full" else None,
        "runs": {},
    }

    for tag, run in runs.items():
        if run["low_bits_root"] is None or run["fp4_matmul_root"] is None:
            results["runs"][tag] = {
                **run,
                "status": "missing_root",
            }
            continue
        if not _has_localcta_runtime_artifacts(run["fp4_matmul_root"]):
            results["runs"][tag] = {
                **run,
                "status": "missing_artifacts",
            }
            continue
        if tag != "current" and run["low_bits_root"] != REPO_ROOT:
            _ensure_historical_te_shim(run["low_bits_root"])
        env = os.environ.copy()
        env["LOW_BITS_TRAINING_ROOT"] = run["low_bits_root"]
        env["FP4_MATMUL_ROOT"] = run["fp4_matmul_root"]
        previous_link = None
        try:
            if run["low_bits_root"] != REPO_ROOT:
                _, previous_link = _set_tmp_fp4_matmul_symlink(run["fp4_matmul_root"])
            proc = subprocess.run(
                base_cmd,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            if run["low_bits_root"] != REPO_ROOT:
                _restore_tmp_fp4_matmul_symlink(previous_link)
        if proc.returncode != 0:
            results["runs"][tag] = {
                **run,
                "status": "error",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr.strip().splitlines()[-20:],
                "stdout_tail": proc.stdout.strip().splitlines()[-20:],
            }
            continue
        results["runs"][tag] = {
            **run,
            "status": "ok",
            "result": _extract_last_json_blob(proc.stdout),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--flavor", default="1B_legacy", choices=["1B", "1B_legacy", "8B"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--mxfp4-backend-version", choices=["v3", "v4"], default="v4")
    parser.add_argument("--block", choices=["full", "qkv", "wo", "ffn"], default="full")
    parser.add_argument("--isolation-m", type=int, default=65536)
    parser.add_argument("--reference-compare", action="store_true")
    parser.add_argument(
        "--reference-tags",
        nargs="+",
        default=["qkv_fast", "ffn_fast", "current_with_qkv_fast_parent", "current_with_ffn_fast_parent", "current_with_ws_sync_parent"],
        choices=sorted(REFERENCE_PAIRS.keys()),
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 64])
    args = parser.parse_args()

    if args.sweep:
        results = []
        for batch_size in args.batch_sizes:
            for mode in MODES:
                print(
                    f"RUN flavor={args.flavor} block={args.block} batch={batch_size} "
                    f"seq={args.seq_len} mode={mode}",
                    flush=True,
                )
                results.append(
                    run_one(
                        mode,
                        args.flavor,
                        batch_size,
                        args.seq_len,
                        args.warmup,
                        args.steps,
                        args.device_index,
                        args.mxfp4_backend_version,
                        args.block,
                        args.isolation_m,
                    )
                )
        print(json.dumps(results, sort_keys=True))
        return

    if args.mode is None:
        parser.error("--mode is required unless --sweep is used")
    if args.reference_compare:
        print(
            json.dumps(
                run_reference_compare(
                    args.mode,
                    args.flavor,
                    args.batch_size,
                    args.seq_len,
                    args.warmup,
                    args.steps,
                    args.device_index,
                    args.mxfp4_backend_version,
                    args.block,
                    args.isolation_m,
                    args.reference_tags,
                ),
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            run_one(
                args.mode,
                args.flavor,
                args.batch_size,
                args.seq_len,
                args.warmup,
                args.steps,
                args.device_index,
                args.mxfp4_backend_version,
                args.block,
                args.isolation_m,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
