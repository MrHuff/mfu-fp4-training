import ctypes
import importlib.util
import json
import os
import signal
import sys
import types

MODE = sys.argv[1]
CONFIG_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else -1
assert MODE in {"regular", "localcta_current", "localcta_parity", "localcta_onepass"}

BACKEND = "localcta" if MODE.startswith("localcta") else "regular"

os.environ["CYPARI_NO_SIGNALS"] = "1"
os.environ["NVTE_NVFP4_DISABLE_RHT"] = "1"
os.environ["NVTE_NVFP4_DISABLE_2D_QUANTIZATION"] = "1"
os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = "0"
os.environ["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] = "1"
os.environ["NVTE_CUSTOM_QUANT"] = "0"
os.environ["USE_TK_QUANT"] = "1"
os.environ["USE_TK_GEMM"] = "1"
os.environ["FUSED_TE_QUANT"] = "0"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NVTE_CUDA_GRAPHS"] = "0"
os.environ["NVTE_FP4_MULTI_STREAM"] = "0"
os.environ["NVTE_MS"] = "0"
os.environ["NVTE_MULTI_STREAM"] = "0"
os.environ["USE_TK_MS"] = "0"
os.environ["USE_TK_LOCALCTA"] = "1" if BACKEND == "localcta" else "0"
os.environ["USE_TK_LOCALCTA_FUSED"] = "0"
os.environ["USE_TK_LOCALCTA_FUSED_DIRECT"] = "0"

ROOT = "/opt/mfu/EXTERNAL_PATH"
sys.path.insert(0, ROOT)
sys.path.insert(0, "/opt/mfu/EXTERNAL_PATH")

pkg_root = os.path.join(ROOT, "low_bits_training")
quant_root = os.path.join(pkg_root, "quantization")
if "low_bits_training" not in sys.modules:
    pkg = types.ModuleType("low_bits_training")
    pkg.__path__ = [pkg_root]
    sys.modules["low_bits_training"] = pkg
if "low_bits_training.quantization" not in sys.modules:
    quant_pkg = types.ModuleType("low_bits_training.quantization")
    quant_pkg.__path__ = [quant_root]
    sys.modules["low_bits_training.quantization"] = quant_pkg

signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
for dep in ["/usr/local/cuda/lib64/libnvrtc.so", "/usr/local/cuda/lib64/libcudart.so"]:
    if os.path.exists(dep):
        ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)

import torch
import transformer_engine as _te
import transformer_engine.pytorch as te

if "low_bits_training.quantization.mxfp_custom_te_fp4" not in sys.modules:
    stub = types.ModuleType("low_bits_training.quantization.mxfp_custom_te_fp4")

    class BoundRecipeLinear(te.Linear):
        def __init__(self, in_features, out_features, bias=True, params_dtype=torch.bfloat16, recipe=None, device=None):
            super().__init__(in_features, out_features, bias=bias, params_dtype=params_dtype, device=device)
            self.bound_recipe = recipe

    stub.BoundRecipeLinear = BoundRecipeLinear
    sys.modules["low_bits_training.quantization.mxfp_custom_te_fp4"] = stub

te_root = "/opt/mfu/EXTERNAL_PATH"
te_pkg = os.path.join(te_root, "transformer_engine")
if os.path.isdir(te_pkg):
    _te.__path__ = [te_pkg]

_so_path = os.environ.get(
    "TE_FUSED_RMSNORM_EXTENSION",
    os.path.join(
        os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"),
        ),
        "py312_cu130",
        "te_fused_rmsnorm_ext_linear",
        "te_fused_rmsnorm_ext_linear.so",
    ),
)
if os.path.exists(_so_path):
    _spec = importlib.util.spec_from_file_location("te_fused_rmsnorm_ext_linear", _so_path)
    _fused_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fused_mod)
else:
    _fused_mod = None

import low_bits_training.quantization.fused_te_linear as _fte
if _fused_mod is not None:
    _fte._te_fused_ext = _fused_mod

from low_bits_training.quantization.fused_te_linear import _fast_quantize
from low_bits_training.quantization.tk_gemm import (
    _get_tk,
    _get_tk_quant_for_gemm,
    _localcta_grouped_k_dgrad_backend,
    _localcta_grouped_k_dgrad_package,
    _split_weight_col_tensors,
)

signal.signal(signal.SIGINT, signal.default_int_handler)


def bench(fn, warmup=3, steps=8):
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
    try:
        for _ in range(warmup):
            fn()
            torch.cuda.synchronize()
        vals = []
        for _ in range(steps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e))
        vals.sort()
        return float(vals[len(vals) // 2])
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)


class _ColRef:
    __slots__ = ("_tk_col",)

    def __init__(self, col):
        self._tk_col = col


def _parse_group_dim1_result(result):
    fp4_row_list, sc_row_list, sg_per_group, fp4_col_list, sc_col_list, \
        a_fp4_full, a_sc_cat, fp4_col_full, sc_col_cat = result
    return fp4_row_list, sc_row_list, sg_per_group, fp4_col_list, sc_col_list, \
        a_fp4_full, a_sc_cat, fp4_col_full, sc_col_cat


def main():
    torch.manual_seed(42)
    M = 65536
    K = 2048
    qkv_dims = [2048, 512, 512]

    tk = _get_tk()
    tkq = _get_tk_quant_for_gemm()

    dy_cat = torch.randn(M, sum(qkv_dims), dtype=torch.bfloat16, device="cuda") * 0.01
    w_qkv = torch.randn(sum(qkv_dims), K, dtype=torch.bfloat16, device="cuda") * 0.01

    q_group = tkq.tk_group_quantize_for_gemm(w_qkv, qkv_dims)
    if BACKEND == "localcta":
        wc_fp4_cols = q_group[3]
        wc_sc_cols = q_group[4]
        col_sg_cat = q_group[5]
        wc_fp4_col_cat = q_group[8] if len(q_group) > 8 else None
        wc_sc_col_cat = q_group[9] if len(q_group) > 9 else None
        if wc_fp4_col_cat is None or wc_sc_col_cat is None:
            col_fp4_cat = torch.cat(
                [fp4.contiguous().view(torch.uint8) for fp4 in wc_fp4_cols], dim=1
            ).view(torch.float4_e2m1fn_x2)
            col_sc_cat = torch.cat(
                [sc.contiguous().view(torch.uint8) for sc in wc_sc_cols], dim=1
            ).view(torch.float8_e4m3fn)
        else:
            col_fp4_cat = wc_fp4_col_cat
            col_sc_cat = wc_sc_col_cat
        w_col_ref = _ColRef((col_fp4_cat, col_sc_cat, col_sg_cat))
    else:
        wc_fp4_cols = q_group[3]
        wc_sc_cols = q_group[4]
        sg_cat = q_group[6]
        col_fp4_cat = torch.cat(
            [fp4.contiguous().view(torch.uint8) for fp4 in wc_fp4_cols], dim=1
        ).view(torch.float4_e2m1fn_x2)
        col_sc_cat = torch.cat(
            [sc.contiguous().view(torch.uint8) for sc in wc_sc_cols], dim=1
        ).view(torch.float8_e4m3fn)
        w_col_ref = _ColRef((col_fp4_cat, col_sc_cat, sg_cat.float()))

    qkv_B_fp4_list, qkv_B_sc_list, qkv_B_sg_list = _split_weight_col_tensors(
        w_col_ref._tk_col[0], w_col_ref._tk_col[1], w_col_ref._tk_col[2], qkv_dims
    )
    qkv_D_list = [torch.empty(M, K, dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    qkv_D_sum = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")

    if MODE == "regular":
        qkv_quant_result = tkq.tk_group_quantize_dim1_for_gemm(dy_cat, qkv_dims)
        qkv_fp4_row_list, qkv_sc_row_list, qkv_sg_per_group, _, _, \
            qkv_a_fp4_full, _, _, _ = _parse_group_dim1_result(qkv_quant_result)

        def qkv_quant_only():
            tkq.tk_group_quantize_dim1_for_gemm(dy_cat, qkv_dims)

        def qkv_gemm_only():
            tk.nvfp4_batched_gemm_strided(
                qkv_a_fp4_full.view(torch.float4_e2m1fn_x2),
                [sc.contiguous().view(torch.float8_e4m3fn) for sc in qkv_sc_row_list],
                [qkv_sg_per_group[i:i+1].to(torch.float32) for i in range(3)],
                [0, qkv_dims[0] // 2, (qkv_dims[0] + qkv_dims[1]) // 2],
                [n // 2 for n in qkv_dims],
                qkv_B_fp4_list, qkv_B_sc_list, qkv_B_sg_list, qkv_D_list,
            )
            tk.sum3_bf16(qkv_D_list[0], qkv_D_list[1], qkv_D_list[2], qkv_D_sum)

        def qkv_full_chain():
            result = tkq.tk_group_quantize_dim1_for_gemm(dy_cat, qkv_dims)
            _, sc_row_list, sg_per_group, _, _, a_fp4_full, _, _, _ = _parse_group_dim1_result(result)
            tk.nvfp4_batched_gemm_strided(
                a_fp4_full.view(torch.float4_e2m1fn_x2),
                [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list],
                [sg_per_group[i:i+1].to(torch.float32) for i in range(3)],
                [0, qkv_dims[0] // 2, (qkv_dims[0] + qkv_dims[1]) // 2],
                [n // 2 for n in qkv_dims],
                qkv_B_fp4_list, qkv_B_sc_list, qkv_B_sg_list, qkv_D_list,
            )
            tk.sum3_bf16(qkv_D_list[0], qkv_D_list[1], qkv_D_list[2], qkv_D_sum)

        dx_max_abs = 0.0
    elif MODE == "localcta_current":
        qkv_package = _localcta_grouped_k_dgrad_package(dy_cat, qkv_dims)

        def qkv_quant_only():
            _localcta_grouped_k_dgrad_package(dy_cat, qkv_dims)

        def qkv_gemm_only():
            _localcta_grouped_k_dgrad_backend(
                qkv_package, w_col_ref, qkv_dims, dx=qkv_D_sum, prefer_strided=True
            )

        def qkv_full_chain():
            package = _localcta_grouped_k_dgrad_package(dy_cat, qkv_dims)
            _localcta_grouped_k_dgrad_backend(
                package, w_col_ref, qkv_dims, dx=qkv_D_sum, prefer_strided=True
            )

        dx_max_abs = 0.0
    elif MODE == "localcta_parity":
        qkv_quant_result = tkq.tk_group_quantize_dim1_for_gemm(dy_cat, qkv_dims)
        qkv_fp4_row_list, qkv_sc_row_list, qkv_sg_per_group, _, _, \
            qkv_a_fp4_full, _, _, _ = _parse_group_dim1_result(qkv_quant_result)

        def qkv_quant_only():
            tkq.tk_group_quantize_dim1_for_gemm(dy_cat, qkv_dims)

        def qkv_gemm_only():
            tk.nvfp4_batched_gemm_strided(
                qkv_a_fp4_full,
                [sc.contiguous().view(torch.float8_e4m3fn) for sc in qkv_sc_row_list],
                [qkv_sg_per_group[i:i+1] for i in range(3)],
                [0, qkv_dims[0] // 2, (qkv_dims[0] + qkv_dims[1]) // 2],
                [n // 2 for n in qkv_dims],
                qkv_B_fp4_list, qkv_B_sc_list, qkv_B_sg_list, qkv_D_list,
            )
            tk.sum3_bf16(qkv_D_list[0], qkv_D_list[1], qkv_D_list[2], qkv_D_sum)

        def qkv_full_chain():
            result = tkq.tk_group_quantize_dim1_for_gemm(dy_cat, qkv_dims)
            _, sc_row_list, sg_per_group, _, _, a_fp4_full, _, _, _ = _parse_group_dim1_result(result)
            tk.nvfp4_batched_gemm_strided(
                a_fp4_full,
                [sc.contiguous().view(torch.float8_e4m3fn) for sc in sc_row_list],
                [sg_per_group[i:i+1] for i in range(3)],
                [0, qkv_dims[0] // 2, (qkv_dims[0] + qkv_dims[1]) // 2],
                [n // 2 for n in qkv_dims],
                qkv_B_fp4_list, qkv_B_sc_list, qkv_B_sg_list, qkv_D_list,
            )
            tk.sum3_bf16(qkv_D_list[0], qkv_D_list[1], qkv_D_list[2], qkv_D_sum)

        dx_max_abs = 0.0
    else:
        qkv_package = _localcta_grouped_k_dgrad_package(dy_cat, qkv_dims)
        a_sc_list = qkv_package["a_sc_list"]
        a_col_offsets = []
        a_col_widths = []
        fp4_off = 0
        for n_i in qkv_dims:
            fp4_cols = n_i // 2
            a_col_offsets.append(fp4_off)
            a_col_widths.append(fp4_cols)
            fp4_off += fp4_cols

        def qkv_quant_only():
            _localcta_grouped_k_dgrad_package(dy_cat, qkv_dims)

        def qkv_gemm_only():
            tk.nvfp4_split3_dgrad_strided_onepass_gemm(
                qkv_package["a_fp4_full"],
                a_sc_list,
                a_col_offsets,
                a_col_widths,
                qkv_B_fp4_list,
                qkv_B_sc_list,
                qkv_B_sg_list,
                qkv_D_sum,
                CONFIG_IDX,
            )

        def qkv_full_chain():
            package = _localcta_grouped_k_dgrad_package(dy_cat, qkv_dims)
            tk.nvfp4_split3_dgrad_strided_onepass_gemm(
                package["a_fp4_full"],
                package["a_sc_list"],
                a_col_offsets,
                a_col_widths,
                qkv_B_fp4_list,
                qkv_B_sc_list,
                qkv_B_sg_list,
                qkv_D_sum,
                CONFIG_IDX,
            )

        ref_dx = torch.empty_like(qkv_D_sum)
        _localcta_grouped_k_dgrad_backend(
            qkv_package, w_col_ref, qkv_dims, dx=ref_dx, prefer_strided=True
        )
        test_dx = torch.empty_like(qkv_D_sum)
        tk.nvfp4_split3_dgrad_strided_onepass_gemm(
            qkv_package["a_fp4_full"],
            a_sc_list,
            a_col_offsets,
            a_col_widths,
            qkv_B_fp4_list,
            qkv_B_sc_list,
            qkv_B_sg_list,
            test_dx,
            CONFIG_IDX,
        )
        dx_max_abs = float((ref_dx.float() - test_dx.float()).abs().max().item())

    result = {
        "mode": MODE,
        "config_idx": CONFIG_IDX,
        "qkv_quant_only_ms": bench(qkv_quant_only),
        "qkv_gemm_only_ms": bench(qkv_gemm_only),
        "qkv_full_chain_ms": bench(qkv_full_chain),
        "dx_max_abs": dx_max_abs,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
