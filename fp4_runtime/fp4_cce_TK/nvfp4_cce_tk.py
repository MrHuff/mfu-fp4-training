"""
NVFP4 Cross-Entropy using ThunderKittens GEMM.

Replaces cuBLASLt NVFP4 GEMM with TK's nvfp4_gemm kernel.
Quantization also uses TK's nvfp4_quantize kernel.

D(M, N) = A(M, K) @ B(N, K)^T  (TN layout)
"""

import os
import sys
import ctypes
import importlib.util
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist

from fp4_cce_TK.extension_loader import find_existing_extension

from fp4_cce_TK.v4_common import (
    assume_all_valid_full_vocab,
    backward_scale_cuda,
    bf16_logits_cuda,
    direct_loss_and_probs,
    direct_loss_and_probs_target_split,
    direct_loss_and_probs_target_top1_split,
    direct_loss_and_probs_target_top2_split,
    direct_loss_and_probs_target_top4_split,
    direct_loss_and_grad_probs,
    direct_loss_lse_target_topk_split,
    direct_loss_lse_target_topk_split_exact_logits,
    direct_loss_lse_target_topk_split_exact_logits_mxfp4_row,
    direct_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered,
    direct_loss_lse_target_topk_split_exact_logits_mxfp8_row,
    loss_and_probs,
    mxfp8_quant_col,
    mxfp8_quant_row,
    mxfp8_quant_row_col,
    mxfp4_quant_row_mxfp8_col,
    mxfp8_rmsnorm_quant_row_col,
    mxfp8_row_nvfp4_col_tiled_g_cache_target_split,
    mxfp8_tiled_g_cache_target_split,
    mxfp4_tiled_g_cache,
    mxfp4_tiled_g_cache_target_split,
    mxfp4_col_requant_from_row,
    mxfp4_softmax_tail_quant_row_col,
    nvfp4_col_requant_from_mxfp8_row,
    mxfp8_col_requant_from_mxfp8_row,
    nvfp4_staged_p_cache,
    nvfp4_staged_g_cache,
    nvfp4_tiled_g_cache,
    nvfp4_tiled_g_cache_target_split,
    nvfp4_tma_g_cache,
    nvfp4_tma_p_cache,
    nvfp4_tiled_p_cache,
    nvfp4_vocab_parallel_tiled_g_cache,
    replace_target_logits_bf16,
    sparse_correct,
    sparse_correct_target_split,
    sparse_correct_target_top1_split,
    sparse_correct_target_top2_split,
    sparse_correct_target_top4_split,
    sparse_correct_target_top6_split,
    sparse_correct_target_topk_split,
    sparse_correct_target_topk_dC,
    sparse_correct_target_topk_dC_prepared,
    sparse_correct_target_topk_dE,
    softmax_repaired_grad_probs_from_lse,
    softmax_repaired_grad_probs_from_lse_inplace,
    add_compact_target_topk_dC,
    compact_target_topk_dC,
    prepare_sparse_correct_target_topk_dC,
    use_staged_nvfp4_p_cache,
    use_tma_nvfp4_p_cache,
    use_tiled_nvfp4_p_cache,
    valid_mask_count_cuda,
    use_nvfp4_vocab_parallel_direct_g_cache,
    vocab_parallel_loss_and_grad_probs,
)
from fp4_cce_TK.mxfp4_cce_tk import (
    MXFP4Quantized,
    quantize_mxfp4_col_tk,
    quantize_mxfp4_norm_row_and_col_with_output_tk,
    quantize_mxfp4_row_tk,
    quantize_mxfp4_row_and_col_tk,
    tk_mxfp4_gemm,
    tk_mxfp4_gemm_atbt,
)

# ---------------------------------------------------------------------------
# Lazy TK NVFP4 import
# ---------------------------------------------------------------------------
_tk_nvfp4 = None
_tk_mxfp8 = None
_nvfp4_localcta_v4_quant = None
_tk_nvfp4_localcta_gemm_v3 = None
_sparse_repair_streams = {}
_sparse_dC_preparation_streams = {}
_sparse_dC_inflight = {}
_direct_fp8_scale_tensors = {}
_mx_backward_gemm_streams = {}

def _repo_roots():
    here = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    roots = [here]
    env_root = os.environ.get("FP4_MATMUL_ROOT")
    if env_root:
        roots.append(os.path.normpath(env_root))
    extra_roots = os.environ.get("FP4_MATMUL_EXTRA_ROOTS", "")
    roots.extend(
        os.path.normpath(root)
        for root in extra_roots.split(os.pathsep)
        if root
    )
    roots.extend([
        '/opt/mfu/EXTERNAL_PATH',
        '/opt/mfu/EXTERNAL_PATH',
        '/tmp/fp4_matmul_v4_pcache',
        '/opt/mfu/EXTERNAL_PATH',
        '/opt/mfu/EXTERNAL_PATH',
    ])
    deduped = []
    for root in roots:
        if root and root not in deduped:
            deduped.append(root)
    return deduped


def _find_existing_so(label, relpath):
    return find_existing_extension(label, _repo_roots(), relpath)


def _get_tk_nvfp4():
    global _tk_nvfp4
    if _tk_nvfp4 is not None:
        return _tk_nvfp4

    so_name = '_C.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "TK NVFP4 _C.so",
        os.path.join('ThunderKittens', 'kernels', 'gemm', 'nvfp4_b200', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    # Load without polluting sys.modules['_C']
    # Must use '_C' as module name — matches the PyInit__C export in the .so
    # Pre-install our new module object so exec_module doesn't find a stale cache
    old_c = sys.modules.pop('_C', None)
    spec = importlib.util.spec_from_file_location('_C', so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_C'] = mod
    spec.loader.exec_module(mod)
    if old_c is not None:
        sys.modules['_C'] = old_c
    else:
        sys.modules.pop('_C', None)

    _tk_nvfp4 = mod
    return _tk_nvfp4


def _get_tk_mxfp8():
    global _tk_mxfp8
    if _tk_mxfp8 is not None:
        return _tk_mxfp8

    so_name = '_C_mxfp8.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "TK MXFP8 extension",
        os.path.join(
            'ThunderKittens', 'kernels', 'gemm', 'mxfp8_b200', so_name
        ),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    spec = importlib.util.spec_from_file_location('_C_mxfp8', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _tk_mxfp8 = mod
    return _tk_mxfp8


# ---------------------------------------------------------------------------
# Lazy NVFP4 v5 quantizer import (persistent kernel)
# ---------------------------------------------------------------------------
_nvfp4_quant_v5 = None

def _get_nvfp4_quant_v5():
    global _nvfp4_quant_v5
    if _nvfp4_quant_v5 is not None:
        return _nvfp4_quant_v5

    so_name = '_tk_quant_v5.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "nvfp4 v5 quantizer",
        os.path.join('TK_quantisation', 'nvfp4_v5', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    # Preload torch lib dependencies
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_python.so'), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location('_tk_quant_v5', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _nvfp4_quant_v5 = mod
    return _nvfp4_quant_v5


def _get_nvfp4_localcta_v4_quant():
    global _nvfp4_localcta_v4_quant
    if _nvfp4_localcta_v4_quant is not None:
        return _nvfp4_localcta_v4_quant

    so_name = '_tk_quant_localcta_v4.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "nvfp4 localCTA v4 quantizer",
        os.path.join('TK_quantisation', 'nvfp4_CTA_local_v4', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_python.so'), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location('_tk_quant_localcta_v4', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _nvfp4_localcta_v4_quant = mod
    return _nvfp4_localcta_v4_quant


def _get_tk_nvfp4_localcta_gemm_v3():
    global _tk_nvfp4_localcta_gemm_v3
    if _tk_nvfp4_localcta_gemm_v3 is not None:
        return _tk_nvfp4_localcta_gemm_v3

    so_name = '_C_nv_localcta_gemm_v3.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "NVFP4 localCTA GEMM v3",
        os.path.join(
            'ThunderKittens',
            'kernels',
            'gemm',
            'nvfp4_b200',
            'localCTA_epilogue_v3',
            so_name,
        ),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_python.so'), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location('_C_nv_localcta_gemm_v3', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _tk_nvfp4_localcta_gemm_v3 = mod
    return _tk_nvfp4_localcta_gemm_v3


def _use_nvfp4_localcta_v4_quant() -> bool:
    backend = os.environ.get("FP4_CCE_V4_NVFP4_QUANT_BACKEND", "").strip().lower()
    if backend in {"localcta", "localcta_v4", "localcta-v4", "v4"}:
        return True
    return os.environ.get("FP4_CCE_V4_NVFP4_LOCALCTA", "0") == "1"


def _use_nvfp4_x_four_over_six_mae() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_X_FOUROVERSIX_MAE", "0") == "1"


def _use_nvfp4_w_four_over_six_mae() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_W_FOUROVERSIX_MAE", "0") == "1"


def _nvfp4_localcta_quant_fn():
    mod = _get_nvfp4_localcta_v4_quant()
    name = os.environ.get(
        "FP4_CCE_V4_NVFP4_LOCALCTA_QUANT_FN",
        "final_sg",
    ).strip().lower()
    mapping = {
        "base": "tk_localcta_quantize_for_gemm",
        "strict": "tk_localcta_quantize_for_gemm",
        "final_sg": "tk_localcta_quantize_for_gemm_final_sg",
        "final-sg": "tk_localcta_quantize_for_gemm_final_sg",
        "opt": "tk_localcta_quantize_for_gemm_opt",
        "final_sg_opt": "tk_localcta_quantize_for_gemm_final_sg_opt",
        "final-sg-opt": "tk_localcta_quantize_for_gemm_final_sg_opt",
        "prepared": "tk_localcta_quantize_for_gemm_prepared",
        "prepared_legacy": "tk_localcta_quantize_for_gemm_prepared",
        "prepared-legacy": "tk_localcta_quantize_for_gemm_prepared",
    }
    fn_name = mapping.get(name)
    if fn_name is None:
        raise ValueError(
            "FP4_CCE_V4_NVFP4_LOCALCTA_QUANT_FN must be one of "
            "{'base','final_sg','opt','final_sg_opt','prepared_legacy'}."
        )
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"localCTA v4 quantizer is missing {fn_name}.")
    return fn


def _quantize_nvfp4_row_and_col_localcta_v4(
    x: torch.Tensor,
    encode_centric: bool = True,
    *,
    four_over_six_mae: bool = False,
    data_stochastic_rounding: bool = False,
    scale_stochastic_rounding: bool = False,
    rng_seed: int = 0,
    rng_subsequence: int = 0,
):
    mode = os.environ.get(
        "FP4_CCE_V4_NVFP4_LOCALCTA_QUANT_FN",
        "final_sg",
    ).strip().lower()
    use_opt = (
        four_over_six_mae
        or data_stochastic_rounding
        or scale_stochastic_rounding
    )
    if use_opt:
        if mode not in {"final_sg", "final-sg", "final_sg_opt", "final-sg-opt"}:
            raise RuntimeError(
                "localCTA optimized quantization requires final_sg or "
                "final_sg_opt mode"
            )
        if four_over_six_mae and (
            data_stochastic_rounding or scale_stochastic_rounding
        ):
            raise RuntimeError(
                "localCTA FourOverSix MAE cannot be combined with stochastic "
                "rounding"
            )
        mod = _get_nvfp4_localcta_v4_quant()
        fn = getattr(mod, "tk_localcta_quantize_for_gemm_final_sg_opt", None)
        if fn is None:
            raise RuntimeError(
                "localCTA v4 quantizer is missing "
                "tk_localcta_quantize_for_gemm_final_sg_opt."
            )
        result = fn(
            x,
            True,
            bool(encode_centric),
            bool(data_stochastic_rounding),
            bool(scale_stochastic_rounding),
            "none",
            False,
            int(rng_seed),
            int(rng_subsequence),
            bool(four_over_six_mae),
        )
    else:
        result = _nvfp4_localcta_quant_fn()(x, True, bool(encode_centric))
    row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg = result[:6]
    keepalive = result[6:]
    if mode in {"prepared", "prepared_legacy", "prepared-legacy"}:
        unit_sg = torch.ones(1, dtype=torch.float32, device=x.device)
        row_sg = unit_sg
        col_sg = unit_sg
        keepalive = (*keepalive, result[4], result[5])
    return (
        NVFP4Quantized(
            row_fp4, row_sc, row_sg, keepalive=keepalive, layout="localcta"
        ),
        NVFP4Quantized(
            col_fp4, col_sc, col_sg, keepalive=keepalive, layout="localcta"
        ),
    )


def quantize_nvfp4_col_localcta_v4(
    x: torch.Tensor,
    encode_centric: bool = True,
    *,
    four_over_six_mae: bool = False,
) -> "NVFP4Quantized":
    """Produce only the transposed localCTA operand needed by backward."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("localCTA column input must be contiguous BF16 [M, K]")
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_quantize_col_for_gemm_final_sg_opt",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the column-only final-SG producer"
        )
    col_fp4, col_sc, col_sg = fn(
        x,
        bool(encode_centric),
        bool(four_over_six_mae),
    )
    return NVFP4Quantized(
        col_fp4,
        col_sc,
        col_sg,
        layout="localcta",
    )


def quantize_nvfp4_row_localcta_v4(
    x: torch.Tensor,
    encode_centric: bool = True,
    *,
    four_over_six_mae: bool = False,
) -> "NVFP4Quantized":
    """Produce only the row operand used by the localCTA forward GEMM."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("localCTA row input must be contiguous BF16 [M, K]")
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_quantize_for_gemm_final_sg_opt",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the row-only final-SG producer"
        )
    result = fn(
        x,
        False,
        bool(encode_centric),
        False,
        False,
        "none",
        False,
        0,
        0,
        bool(four_over_six_mae),
    )
    row_fp4, row_sc, _col_fp4, _col_sc, row_sg, _col_sg = result[:6]
    return NVFP4Quantized(
        row_fp4,
        row_sc,
        row_sg,
        keepalive=result[6:],
        layout="localcta",
    )


def quantize_nvfp4_xw_row_and_col_tk(
    x: torch.Tensor,
    encode_centric: bool = True,
    *,
    role: str | None = None,
):
    if role not in {None, "X", "W"}:
        raise ValueError(f"NVFP4 head quantization role must be X or W, got {role!r}")
    four_over_six_mae = (
        role == "X" and _use_nvfp4_x_four_over_six_mae()
    ) or (
        role == "W" and _use_nvfp4_w_four_over_six_mae()
    )
    weight_data_sr = (
        role == "W"
        and os.environ.get("FP4_CCE_V4_NVFP4_W_DATA_SR", "0") != "0"
    )
    weight_scale_sr = (
        role == "W"
        and os.environ.get("FP4_CCE_V4_NVFP4_W_SCALE_SR", "0") != "0"
    )
    weight_sr_enabled = weight_data_sr or weight_scale_sr
    if four_over_six_mae or weight_data_sr or weight_scale_sr:
        if not _use_nvfp4_localcta_v4_quant():
            raise RuntimeError(
                f"NVFP4 {role} FourOverSix/SR policy requires the localCTA v4 "
                "quantizer"
            )
        return _quantize_nvfp4_row_and_col_localcta_v4(
            x,
            encode_centric=encode_centric,
            four_over_six_mae=four_over_six_mae,
            data_stochastic_rounding=weight_data_sr,
            scale_stochastic_rounding=weight_scale_sr,
            rng_seed=(
                int(os.environ.get("FP4_CCE_V4_NVFP4_W_SR_SEED", "0"))
                if weight_sr_enabled
                else 0
            ),
            rng_subsequence=(
                int(
                    os.environ.get(
                        "FP4_CCE_V4_NVFP4_W_SR_SUBSEQUENCE", "0"
                    )
                )
                if weight_sr_enabled
                else 0
            ),
        )
    return quantize_nvfp4_row_and_col_tk(
        x,
        encode_centric=encode_centric,
    )


def _use_nvfp4_localcta_gemm(A_q=None, B_q=None) -> bool:
    if not _use_nvfp4_localcta_v4_quant():
        return False
    if A_q is not None and B_q is not None:
        a_localcta = A_q.layout == "localcta"
        b_localcta = B_q.layout == "localcta"
        if a_localcta != b_localcta:
            raise RuntimeError(
                "NVFP4 GEMM operands mix localCTA and regular-TK scale layouts: "
                f"A={A_q.layout} sc={tuple(A_q.sc.shape)} sg={tuple(A_q.sg.shape)}, "
                f"B={B_q.layout} sc={tuple(B_q.sc.shape)} sg={tuple(B_q.sg.shape)}"
            )
        if not a_localcta:
            return False
    override = os.environ.get("FP4_CCE_V4_NVFP4_LOCALCTA_GEMM")
    if override is not None:
        return override != "0"
    mode = os.environ.get("FP4_CCE_V4_NVFP4_LOCALCTA_QUANT_FN", "final_sg").strip().lower()
    if mode in {"prepared", "prepared_legacy", "prepared-legacy"}:
        return False
    if A_q is not None and B_q is not None:
        return True
    return True


# ---------------------------------------------------------------------------
# Lazy NVFP4 CCE backward kernel import
# ---------------------------------------------------------------------------
_nvfp4_cce_backward = None

def _get_nvfp4_cce_backward():
    global _nvfp4_cce_backward
    if _nvfp4_cce_backward is not None:
        return _nvfp4_cce_backward

    so_name = '_C_nv_cce_backward.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "NVFP4 CCE backward .so",
        os.path.join('ThunderKittens', 'kernels', 'gemm', 'nvfp4_b200', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_python.so'), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location('_C_nv_cce_backward', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _nvfp4_cce_backward = mod
    return _nvfp4_cce_backward


# ---------------------------------------------------------------------------
# NVFP4 Quantization via TK v5
# ---------------------------------------------------------------------------
class NVFP4Quantized:
    __slots__ = ['fp4', 'sc', 'sg', 'bf16', 'layout', '_keepalive']

    def __init__(self, fp4, sc, sg, bf16=None, keepalive=(), layout=None):
        self.fp4 = fp4    # [M, K//2] fp4x2
        self.sc = sc      # [M//128, K//64, 512] fp8_e4m3fn
        self.sg = sg      # [1] float32
        self.bf16 = bf16  # original bf16 data (for backward)
        self.layout = layout or (
            "localcta" if sg.dim() > 1 or sg.numel() != 1 else "tk"
        )
        self._keepalive = tuple(keepalive)


class MXFP8Quantized:
    __slots__ = ['fp8', 'sc']

    def __init__(self, fp8, sc):
        self.fp8 = fp8
        self.sc = sc


def _empty_mxfp8_quantized(device: torch.device) -> MXFP8Quantized:
    """Return a typed sentinel for a deliberately omitted MXFP8 operand."""
    return MXFP8Quantized(
        torch.empty(0, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(0, dtype=torch.uint8, device=device),
    )


def _empty_mxfp8_backward_cache_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Return empty save-schema slots for G-row/G-col/X-col/W-col caches."""
    slots = []
    for _ in range(4):
        operand = _empty_mxfp8_quantized(device)
        slots.extend(
            (
                operand.fp8,
                operand.sc,
                torch.empty(0, dtype=torch.float32, device=device),
            )
        )
    return tuple(slots)


class MXFP6Quantized:
    __slots__ = ['fp6', 'sc', 'format']

    def __init__(self, fp6, sc, format="e2m3"):
        if format not in {"e2m3", "e3m2"}:
            raise ValueError(f"unsupported MXFP6 format: {format}")
        self.fp6 = fp6
        self.sc = sc
        self.format = format


class DirectFP8Quantized:
    __slots__ = ["fp8", "sc"]

    def __init__(self, fp8, scale):
        self.fp8 = fp8
        self.sc = scale


def _direct_fp8_scale_tensor(device: torch.device, scale: float):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (index, scale)
    tensor = _direct_fp8_scale_tensors.get(key)
    if tensor is None:
        tensor = torch.tensor(
            scale, dtype=torch.float32, device=torch.device("cuda", index)
        )
        _direct_fp8_scale_tensors[key] = tensor
    return tensor


def quantize_direct_fp8_row_mxfp4_col(
    x: torch.Tensor,
    *,
    role: str,
) -> tuple[DirectFP8Quantized, MXFP4Quantized]:
    """Experimental fixed-scale E4M3 row plus MXFP4 backward column."""
    if role not in ("X", "W"):
        raise ValueError("direct FP8 role must be X or W")
    scale = float(
        os.environ.get(
            f"FP4_CCE_V4_DIRECT_FP8_{role}_SCALE",
            "0.0625" if role == "X" else "0.00048828125",
        )
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("direct FP8 scale must be finite and positive")
    scale_tensor = _direct_fp8_scale_tensor(x.device, scale)
    producer = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_quantize_direct_fp8_row_mxfp4_col",
        None,
    )
    if (
        producer is not None
        and os.environ.get(
            "FP4_CCE_V4_DIRECT_FP8_FUSED_PRODUCER", "1"
        ) != "0"
    ):
        row_fp8, col_fp4, col_sc = producer(x, scale)
        return (
            DirectFP8Quantized(row_fp8, scale_tensor),
            MXFP4Quantized(col_fp4, col_sc),
        )

    row_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return (
        DirectFP8Quantized(row_fp8, scale_tensor),
        quantize_mxfp4_col_tk(x, mode=1),
    )


def quantize_mxfp8_tk(x: torch.Tensor) -> MXFP8Quantized:
    """Quantize contiguous BF16 [M, K] into native TK MXFP8 layout."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("MXFP8 input must be contiguous BF16 [M, K]")
    M, K = x.shape
    if M % 128 or K % 128:
        raise ValueError(
            f"MXFP8 dimensions must be multiples of 128, got M={M}, K={K}"
        )
    fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sc = torch.empty(
        (M // 128, K // 128, 32, 16),
        dtype=torch.uint8,
        device=x.device,
    )
    _get_tk_mxfp8().mxfp8_quantize(x, fp8, sc)
    return MXFP8Quantized(fp8, sc)


def quantize_mxfp6_tk(
    x: torch.Tensor,
    *,
    format: str = "e2m3",
) -> MXFP6Quantized:
    """Quantize contiguous BF16 [M, K] into packed native TK MXFP6."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("MXFP6 input must be contiguous BF16 [M, K]")
    M, K = x.shape
    if M % 128 or K % 128:
        raise ValueError(
            f"MXFP6 dimensions must be multiples of 128, got M={M}, K={K}"
        )
    if format not in {"e2m3", "e3m2"}:
        raise ValueError(f"unsupported MXFP6 format: {format}")
    packed = torch.empty((M, K * 3 // 4), dtype=torch.uint8, device=x.device)
    sc = torch.empty(
        (M // 128, K // 128, 32, 16),
        dtype=torch.uint8,
        device=x.device,
    )
    extension = _get_tk_mxfp8()
    getattr(extension, f"mxfp6_{format}_quantize")(x, packed, sc)
    return MXFP6Quantized(packed, sc, format=format)


def quantize_mxfp6_row_mxfp4_col(
    x: torch.Tensor,
    *,
    format: str = "e2m3",
) -> tuple[MXFP6Quantized, MXFP4Quantized]:
    """Produce an MXFP6 forward row and MXFP4 backward column operand."""
    return quantize_mxfp6_tk(x, format=format), quantize_mxfp4_col_tk(
        x, mode=1
    )


def quantize_mxfp8_row_only_localcta_v4(
    x: torch.Tensor,
) -> MXFP8Quantized:
    """Produce the deployed localCTA MXFP8 row without a transpose operand."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("MXFP8 row-only input must be contiguous BF16 [M, K]")
    M, K = x.shape
    if M % 128 or K % 128:
        raise ValueError(
            f"MXFP8 row-only dimensions must be multiples of 128, got M={M}, K={K}"
        )
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_quantize_mxfp8_row_only",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the MXFP8 row-only producer"
        )
    row_fp8, row_sc = fn(x)
    return MXFP8Quantized(row_fp8, row_sc)


def quantize_mxfp8_row_and_col_fused(
    x: torch.Tensor,
    *,
    quant_max: float = 448.0,
) -> tuple[MXFP8Quantized, MXFP8Quantized]:
    """Quantize contiguous BF16 ``x`` and ``x.T`` in one tiled pass."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("fused MXFP8 row/col input must be contiguous BF16 [M, K]")
    M, K = x.shape
    if M % 128 or K % 128:
        raise ValueError(
            f"fused MXFP8 dimensions must be multiples of 128, got M={M}, K={K}"
        )
    if _use_lowp_logits_bf16_both_inplace_g():
        if float(quant_max) != 448.0:
            raise RuntimeError(
                "BF16-both cache elision requires the deployed MXFP8 quant_max=448"
            )
        row_fp8, row_sc = mxfp8_quant_row(x, float(quant_max))
        return (
            MXFP8Quantized(row_fp8, row_sc),
            _empty_mxfp8_quantized(x.device),
        )
    row_fp8, row_sc, col_fp8, col_sc = mxfp8_quant_row_col(
        x,
        float(quant_max),
    )
    return MXFP8Quantized(row_fp8, row_sc), MXFP8Quantized(col_fp8, col_sc)


def quantize_mxfp8_row_nvfp4_col_localcta_v4(
    x: torch.Tensor,
    encode_centric: bool = True,
    *,
    four_over_six_mae: bool = False,
) -> tuple[MXFP8Quantized, NVFP4Quantized | MXFP8Quantized]:
    """Produce MXFP8 forward and localCTA NVFP4 backward operands together."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("fused MXFP8/NVFP4 input must be contiguous BF16 [M, K]")
    if _use_mxfp8_g_cache():
        return quantize_mxfp8_row_and_col_fused(x)
    if _use_mixed_dw_mxfp8_cols():
        row = quantize_mxfp8_row_only_localcta_v4(x)
        if _use_lowp_logits_bf16_both_inplace_g():
            return row, _empty_mxfp8_quantized(x.device)
        col_fp8, col_sc = mxfp8_quant_col(x)
        return (
            row,
            MXFP8Quantized(col_fp8, col_sc),
        )
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_quantize_mxfp8_row_nvfp4_col_final_sg_opt",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the fused MXFP8/NVFP4 producer"
        )
    col_data_sr = _env_flag(
        "FP4_CCE_V4_NVFP4_X_COL_DATA_SR", False
    )
    row_fp8, row_mxsc, col_fp4, col_sc, col_sg = fn(
        x,
        bool(encode_centric),
        bool(four_over_six_mae),
        col_data_sr,
        int(os.environ.get("FP4_CCE_V4_NVFP4_X_COL_SR_SEED", "0")),
        int(
            os.environ.get(
                "FP4_CCE_V4_NVFP4_X_COL_SR_SUBSEQUENCE", "768"
            )
        )
        + int(os.environ.get("RANK", "0")),
    )
    return (
        MXFP8Quantized(row_fp8, row_mxsc),
        NVFP4Quantized(col_fp4, col_sc, col_sg, layout="localcta"),
    )


def quantize_mxfp8_row_mxfp4_col(
    x: torch.Tensor,
) -> tuple[MXFP8Quantized, MXFP4Quantized]:
    """Produce an MXFP8 forward operand and an MXFP4 backward column."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("fused MXFP8/MXFP4 input must be contiguous BF16 [M, K]")
    if x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError(
            "fused MXFP8/MXFP4 dimensions must be multiples of 128, "
            f"got M={x.shape[0]}, K={x.shape[1]}"
        )
    producer_mode = os.environ.get(
        "FP4_CCE_V4_MXFP4_FUSED_INPUT_PRODUCER", "auto"
    ).strip().lower()
    if producer_mode not in {"0", "1", "auto"}:
        raise ValueError(
            "FP4_CCE_V4_MXFP4_FUSED_INPUT_PRODUCER must be 0, 1, or auto"
        )
    # The single-load CTA wins for activation matrices but the two mature TMA
    # producers still edge it on very tall vocabulary weights.
    if producer_mode == "0" or (
        producer_mode == "auto" and x.shape[0] > 8192
    ):
        return quantize_mxfp8_tk(x), quantize_mxfp4_col_tk(x, mode=1)
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_quantize_mxfp8_row_mxfp4_col",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the fused MXFP8/MXFP4 producer"
        )
    row_fp8, row_sc, col_fp4, col_sc = fn(x)
    return MXFP8Quantized(row_fp8, row_sc), MXFP4Quantized(col_fp4, col_sc)


def quantize_mxfp4_row_nvfp4_col_v5(
    x: torch.Tensor,
    encode_centric: bool = True,
    role: str | None = None,
) -> tuple[MXFP4Quantized, NVFP4Quantized]:
    """Produce an MXFP4 row and native-v5 NVFP4 column in two shared passes."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("mixed MXFP4/NVFP4 input must be contiguous BF16 [M, K]")
    if x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError(
            "mixed MXFP4/NVFP4 dimensions must be multiples of 128, "
            f"got M={x.shape[0]}, K={x.shape[1]}"
        )
    if not encode_centric:
        raise ValueError("mixed MXFP4/NVFP4 producer requires encode-centric NVFP4")
    if role not in (None, "X", "W"):
        raise ValueError("mixed MXFP4/NVFP4 role must be X, W, or None")

    data_sr = _env_flag("FP4_CCE_V4_MXFP4_FORWARD_DATA_SR", False)
    if role is not None:
        role_data_sr = os.environ.get(
            f"FP4_CCE_V4_MXFP4_FORWARD_{role}_DATA_SR"
        )
        if role_data_sr is not None:
            data_sr = role_data_sr != "0"

    v5 = _get_nvfp4_quant_v5()
    producer = getattr(v5, "tk_quantize_mxfp4_row_nvfp4_col", None)
    if producer is None:
        if data_sr:
            raise RuntimeError(
                "MXFP4 forward stochastic rounding requires the fused "
                "MXFP4-row/native-v5-column producer"
            )
        row = quantize_mxfp4_row_tk(x, mode=1)
        _unused_row, col = _quantize_nvfp4_row_and_col_v5(
            x, encode_centric=True
        )
        return row, col

    threads = int(os.environ.get("FP4_CCE_V4_MIXED_PRODUCER_THREADS", "128"))
    rank = int(os.environ.get("RANK", "0"))
    if rank < 0:
        raise ValueError("distributed rank must be non-negative")
    rng_seed = int(
        os.environ.get("FP4_CCE_V4_MXFP4_FORWARD_RNG_SEED", "0")
    )
    rng_seed ^= (rank * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    if role == "W":
        rng_seed ^= 0xD1B54A32D192ED03
    rng_seed &= (1 << 64) - 1
    rng_subsequence = int(
        os.environ.get("FP4_CCE_V4_MXFP4_FORWARD_RNG_SUBSEQUENCE_BASE", "0")
    )
    row_fp4, row_sc, col_fp4, col_sc, col_sg = producer(
        x,
        threads,
        data_sr,
        rng_seed,
        rng_subsequence,
    )
    return (
        MXFP4Quantized(row_fp4, row_sc),
        NVFP4Quantized(col_fp4, col_sc, col_sg),
    )


def quantize_mxfp4_row_mxfp8_col(
    x: torch.Tensor,
) -> tuple[MXFP4Quantized, MXFP8Quantized]:
    """Produce an MXFP4 row and an MXFP8 operand for ``x.T`` in one pass."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("mixed MXFP4/MXFP8 input must be contiguous BF16 [M, K]")
    if x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError(
            "mixed MXFP4/MXFP8 dimensions must be multiples of 128, "
            f"got M={x.shape[0]}, K={x.shape[1]}"
        )
    row_fp4, row_sc, col_fp8, col_sc = mxfp4_quant_row_mxfp8_col(x)
    return MXFP4Quantized(row_fp4, row_sc), MXFP8Quantized(col_fp8, col_sc)


def quantize_mxfp8_norm_row_mxfp4_col_with_output_localcta_v4(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
):
    """Fuse RMSNorm output, MXFP8 row, and MXFP4 transpose production."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("fused RMSNorm MXFP8/MXFP4 input must be contiguous BF16 [M, K]")
    if gamma.ndim != 1 or gamma.dtype != torch.bfloat16 or not gamma.is_contiguous():
        raise ValueError("fused RMSNorm MXFP8/MXFP4 gamma must be contiguous BF16 [K]")
    if x.shape[1] != gamma.shape[0]:
        raise ValueError("RMSNorm gamma must match the input K dimension")
    if x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError(
            "fused RMSNorm MXFP8/MXFP4 dimensions must be multiples of 128, "
            f"got M={x.shape[0]}, K={x.shape[1]}"
        )
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_rmsnorm_quantize_mxfp8_row_mxfp4_col_with_output",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the fused RMSNorm MXFP8/MXFP4 producer"
        )
    normed, row_fp8, row_sc, col_fp4, col_sc, inv_rms = fn(
        x, gamma, float(epsilon)
    )
    return (
        normed,
        MXFP8Quantized(row_fp8, row_sc),
        MXFP4Quantized(col_fp4, col_sc),
        inv_rms,
        torch.empty(0, dtype=torch.float32, device=x.device),
    )


def quantize_direct_fp8_norm_row_mxfp4_col_with_output_localcta_v4(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    *,
    role: str = "X",
):
    """Fuse RMSNorm output, fixed-scale E4M3 row, and MXFP4 transpose."""
    if role not in ("X", "W"):
        raise ValueError("direct FP8 role must be X or W")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("fused RMSNorm direct-FP8/MXFP4 input must be contiguous BF16 [M, K]")
    if gamma.ndim != 1 or gamma.dtype != torch.bfloat16 or not gamma.is_contiguous():
        raise ValueError("fused RMSNorm direct-FP8/MXFP4 gamma must be contiguous BF16 [K]")
    if x.shape[1] != gamma.shape[0]:
        raise ValueError("RMSNorm gamma must match the input K dimension")
    if x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError(
            "fused RMSNorm direct-FP8/MXFP4 dimensions must be multiples of 128, "
            f"got M={x.shape[0]}, K={x.shape[1]}"
        )
    scale = float(
        os.environ.get(
            f"FP4_CCE_V4_DIRECT_FP8_{role}_SCALE",
            "0.0625" if role == "X" else "0.00048828125",
        )
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("direct FP8 scale must be finite and positive")
    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_rmsnorm_quantize_direct_fp8_row_mxfp4_col_with_output",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the fused RMSNorm direct-FP8/MXFP4 producer"
        )
    normed, row_fp8, _row_sc, col_fp4, col_sc, inv_rms = fn(
        x, gamma, float(epsilon), scale
    )
    return (
        normed,
        DirectFP8Quantized(row_fp8, _direct_fp8_scale_tensor(x.device, scale)),
        MXFP4Quantized(col_fp4, col_sc),
        inv_rms,
        torch.empty(0, dtype=torch.float32, device=x.device),
    )


def quantize_mxfp8_norm_row_nvfp4_col_with_output_localcta_v4(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    with_silu: bool = False,
    encode_centric: bool = True,
    *,
    four_over_six_mae: bool = False,
):
    """Fuse RMSNorm output, MXFP8 forward, and localCTA NVFP4 backward production."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("fused RMSNorm MXFP8/NVFP4 input must be contiguous BF16 [M, K]")
    if gamma.ndim != 1 or gamma.dtype != torch.bfloat16 or not gamma.is_contiguous():
        raise ValueError("fused RMSNorm MXFP8/NVFP4 gamma must be contiguous BF16 [K]")
    if x.shape[1] != gamma.shape[0]:
        raise ValueError("RMSNorm gamma must match the input K dimension")
    if x.shape[0] % 128 or x.shape[1] % 128:
        raise ValueError(
            "fused RMSNorm MXFP8/NVFP4 dimensions must be multiples of 128, "
            f"got M={x.shape[0]}, K={x.shape[1]}"
        )
    if with_silu:
        raise ValueError("fused RMSNorm MXFP8/NVFP4 producer does not support SiLU")

    if _use_mxfp8_g_cache():
        normed, row_fp8, row_sc, col_fp8, col_sc, inv_rms = (
            mxfp8_rmsnorm_quant_row_col(
                x,
                gamma,
                float(epsilon),
                448.0,
            )
        )
        return (
            normed,
            MXFP8Quantized(row_fp8, row_sc),
            MXFP8Quantized(col_fp8, col_sc),
            inv_rms,
            torch.empty(0, dtype=torch.float32, device=x.device),
        )

    if _use_mixed_dw_mxfp8_cols():
        fn = getattr(
            _get_nvfp4_localcta_v4_quant(),
            "tk_localcta_rmsnorm_quantize_mxfp8_row_with_output",
            None,
        )
        if fn is None:
            raise RuntimeError(
                "localCTA v4 quantizer is missing the fused RMSNorm MXFP8 "
                "row-only producer"
            )
        normed, row_fp8, row_sc, inv_rms = fn(
            x,
            gamma,
            float(epsilon),
        )
        if _use_lowp_logits_bf16_both_inplace_g():
            return (
                normed,
                MXFP8Quantized(row_fp8, row_sc),
                _empty_mxfp8_quantized(x.device),
                inv_rms,
                torch.empty(0, dtype=torch.float32, device=x.device),
            )
        col_fp8, col_sc = mxfp8_quant_col(normed)
        return (
            normed,
            MXFP8Quantized(row_fp8, row_sc),
            MXFP8Quantized(col_fp8, col_sc),
            inv_rms,
            torch.empty(0, dtype=torch.float32, device=x.device),
        )

    fn = getattr(
        _get_nvfp4_localcta_v4_quant(),
        "tk_localcta_rmsnorm_quantize_mxfp8_row_nvfp4_col_with_output_final_sg_opt",
        None,
    )
    if fn is None:
        raise RuntimeError(
            "localCTA v4 quantizer is missing the fused RMSNorm MXFP8/NVFP4 producer"
        )
    col_data_sr = _env_flag(
        "FP4_CCE_V4_NVFP4_X_COL_DATA_SR", False
    )
    normed, row_fp8, row_mxsc, col_fp4, col_sc, col_sg, inv_rms = fn(
        x,
        gamma,
        float(epsilon),
        bool(encode_centric),
        bool(four_over_six_mae),
        col_data_sr,
        int(os.environ.get("FP4_CCE_V4_NVFP4_X_COL_SR_SEED", "0")),
        int(
            os.environ.get(
                "FP4_CCE_V4_NVFP4_X_COL_SR_SUBSEQUENCE", "768"
            )
        )
        + int(os.environ.get("RANK", "0")),
    )
    return (
        normed,
        MXFP8Quantized(row_fp8, row_mxsc),
        NVFP4Quantized(col_fp4, col_sc, col_sg, layout="localcta"),
        inv_rms,
        torch.empty(0, dtype=torch.float32, device=x.device),
    )


def tk_mxfp8_gemm(
    A_q: MXFP8Quantized,
    B_q: MXFP8Quantized,
    out: torch.Tensor | None = None,
    *,
    fp8_output: bool = False,
    output_scale: torch.Tensor | None = None,
    config_env_name: str = "FP4_CCE_V4_MXFP8_GEMM_CONFIG",
    config_override: str | int | None = None,
) -> torch.Tensor:
    """Compute A @ B.T from native TK MXFP8 operands.

    ``output_scale`` selects the optional fused BF16 scaling epilogue.  The
    extension deliberately mirrors ``BF16_output.mul_(CUDA_FP32_scalar)``
    bit-for-bit; it is currently used only by the backward-only MXFP8 dWeight
    analysis path.
    """
    M, K = A_q.fp8.shape
    N, K_b = B_q.fp8.shape
    if K_b != K:
        raise ValueError(f"MXFP8 GEMM K mismatch: {K} != {K_b}")
    if out is None:
        out = torch.empty(
            (M, N),
            dtype=torch.float8_e4m3fn if fp8_output else torch.bfloat16,
            device=A_q.fp8.device,
        )
    expected_dtype = torch.float8_e4m3fn if fp8_output else torch.bfloat16
    if out.dtype != expected_dtype:
        raise ValueError(
            f"MXFP8 GEMM output must use {expected_dtype}, got {out.dtype}"
        )
    if output_scale is not None and fp8_output:
        raise ValueError("scaled MXFP8 GEMM does not support FP8 output")
    extension = _get_tk_mxfp8()
    config_raw = str(
        os.environ.get(config_env_name, "default")
        if config_override is None
        else config_override
    ).strip().lower()
    if fp8_output or config_raw in {"", "default", "none", "off", "-1"}:
        if fp8_output:
            gemm = extension.mxfp8_gemm_fp8
        elif output_scale is not None:
            gemm = getattr(extension, "mxfp8_gemm_scaled", None)
            if gemm is None:
                raise RuntimeError(
                    "MXFP8 extension is missing mxfp8_gemm_scaled; use the "
                    "pinned scaled-epilogue build for this analysis path"
                )
        else:
            gemm = extension.mxfp8_gemm
        config = None
    else:
        config = int(config_raw)
        if config not in range(24):
            raise ValueError(
                f"{config_env_name} must be default or 0-23"
            )
        if output_scale is not None:
            gemm = getattr(extension, "mxfp8_gemm_scaled_config", None)
            if gemm is None:
                raise RuntimeError(
                    "MXFP8 extension is missing mxfp8_gemm_scaled_config; "
                    "use the pinned scaled-epilogue build for this analysis "
                    "path"
                )
        else:
            gemm = extension.mxfp8_gemm_config
    gemm(
        A_q.fp8,
        A_q.sc,
        B_q.fp8,
        B_q.sc,
        out,
        *((output_scale,) if output_scale is not None else ()),
        *(() if config is None else (config,)),
    )
    return out


def tk_mxfp8_gemm_centered(
    A_q: MXFP8Quantized,
    B_q: MXFP8Quantized,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute A @ B.T as E4M3 residuals plus one BF16 max per 32 logits."""
    M, K = A_q.fp8.shape
    N, K_b = B_q.fp8.shape
    if K_b != K:
        raise ValueError(f"MXFP8 GEMM K mismatch: {K} != {K_b}")
    if N % 32:
        raise ValueError(
            f"centered MXFP8 output width must be divisible by 32, got {N}"
        )
    residuals = torch.empty(
        (M, N), dtype=torch.float8_e4m3fn, device=A_q.fp8.device
    )
    centers = torch.empty(
        (M, N // 32), dtype=torch.bfloat16, device=A_q.fp8.device
    )
    _get_tk_mxfp8().mxfp8_gemm_centered_fp8(
        A_q.fp8,
        A_q.sc,
        B_q.fp8,
        B_q.sc,
        residuals,
        centers,
    )
    return residuals, centers


def tk_mxfp6_gemm(
    A_q: MXFP6Quantized,
    B_q: MXFP6Quantized,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute A @ B.T from matching native TK MXFP6 operands."""
    if A_q.format != B_q.format:
        raise ValueError(
            f"MXFP6 GEMM format mismatch: {A_q.format} != {B_q.format}"
        )
    M = A_q.fp6.shape[0]
    N = B_q.fp6.shape[0]
    K = A_q.fp6.shape[1] * 4 // 3
    K_b = B_q.fp6.shape[1] * 4 // 3
    if K_b != K:
        raise ValueError(f"MXFP6 GEMM K mismatch: {K} != {K_b}")
    if out is None:
        out = torch.empty(
            (M, N), dtype=torch.bfloat16, device=A_q.fp6.device
        )
    if out.dtype != torch.bfloat16 or out.shape != (M, N):
        raise ValueError(
            f"MXFP6 GEMM output must be BF16 [{M}, {N}], got "
            f"{out.dtype} {tuple(out.shape)}"
        )
    extension = _get_tk_mxfp8()
    config_raw = os.environ.get(
        "FP4_CCE_V4_MXFP6_GEMM_CONFIG", "default"
    ).strip().lower()
    if (
        A_q.format == "e2m3"
        and config_raw not in {"", "default", "none", "off", "-1"}
    ):
        config = int(config_raw)
        if config not in range(15):
            raise ValueError(
                "FP4_CCE_V4_MXFP6_GEMM_CONFIG must be default or 0-14"
            )
        extension.mxfp6_e2m3_gemm_config(
            A_q.fp6, A_q.sc, B_q.fp6, B_q.sc, out, config
        )
    else:
        if A_q.format == "e3m2" and config_raw not in {
            "",
            "default",
            "none",
            "off",
            "-1",
        }:
            raise ValueError("configured MXFP6 GEMM currently supports E2M3 only")
        getattr(extension, f"mxfp6_{A_q.format}_gemm")(
            A_q.fp6, A_q.sc, B_q.fp6, B_q.sc, out
        )
    return out


def direct_fp8_gemm(
    A_q: DirectFP8Quantized,
    B_q: DirectFP8Quantized,
) -> torch.Tensor:
    """Compute A @ B.T from fixed-scale E4M3 operands."""
    out_dtype = (
        torch.float8_e4m3fn
        if _env_flag("FP4_CCE_V4_DIRECT_FP8_LOGITS", False)
        else torch.bfloat16
    )
    return torch._scaled_mm(
        A_q.fp8,
        B_q.fp8.T,
        A_q.sc,
        B_q.sc,
        out_dtype=out_dtype,
        use_fast_accum=True,
    )


def quantize_nvfp4_tk(
    x: torch.Tensor,
    keep_bf16: bool = True,
    encode_centric: bool = True,
) -> NVFP4Quantized:
    """Quantize BF16 (M, K) → NVFP4 using v5 persistent quantizer."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    if _use_nvfp4_localcta_v4_quant():
        row_q, _col_q = _quantize_nvfp4_row_and_col_localcta_v4(
            x,
            encode_centric=encode_centric,
        )
        row_q.bf16 = x if keep_bf16 else None
        return row_q

    v5 = _get_nvfp4_quant_v5()
    result = v5.tk_quantize_for_gemm(
        x, False, bool(encode_centric)
    )
    row_fp4, row_sc, _col_fp4, _col_sc, sg, _sg2 = result[:6]

    return NVFP4Quantized(row_fp4, row_sc, sg, bf16=x if keep_bf16 else None, keepalive=result[6:])


def quantize_nvfp4_row_and_col_tk(
    x: torch.Tensor,
    encode_centric: bool = True,
):
    """Quantize BF16 (M, K) and its transpose for GEMM consumers.

    Returns (row_q, col_q), where col_q represents x.T as GEMM row data.
    """
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    if _use_nvfp4_localcta_v4_quant():
        return _quantize_nvfp4_row_and_col_localcta_v4(
            x,
            encode_centric=encode_centric,
        )

    return _quantize_nvfp4_row_and_col_v5(
        x,
        encode_centric=encode_centric,
    )


def _quantize_nvfp4_row_and_col_v5(
    x: torch.Tensor,
    encode_centric: bool = True,
):
    """Quantize with the regular-TK layout regardless of the X/W policy."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, (
        f"Dims must be multiples of 128, got M={M}, K={K}"
    )

    v5 = _get_nvfp4_quant_v5()
    result = v5.tk_quantize_for_gemm(
        x, True, bool(encode_centric)
    )
    row_fp4, row_sc, col_fp4, col_sc, sg, _sg2 = result[:6]
    keepalive = result[6:]
    return (
        NVFP4Quantized(row_fp4, row_sc, sg, keepalive=keepalive),
        NVFP4Quantized(col_fp4, col_sc, sg, keepalive=keepalive),
    )


def _quantize_nvfp4_row_and_col_v5_sr(
    x: torch.Tensor,
    encode_centric: bool = True,
    *,
    data_stochastic_rounding: bool,
    scale_stochastic_rounding: bool = False,
    rng_seed: int = 0,
    rng_subsequence: int = 0,
):
    """Quantize a backward operand with independently controlled native SR."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, (
        f"Dims must be multiples of 128, got M={M}, K={K}"
    )

    v5 = _get_nvfp4_quant_v5()
    result = v5.tk_quantize_for_gemm_opt(
        x,
        True,
        bool(encode_centric),
        bool(data_stochastic_rounding),
        bool(scale_stochastic_rounding),
        "none",
        False,
        int(rng_seed),
        int(rng_subsequence),
        float(
            os.environ.get(
                "FP4_CCE_V4_NVFP4_BACKWARD_SCALE_TARGET", "448"
            )
        ),
    )
    row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg = result[:6]
    keepalive = result[6:]
    return (
        NVFP4Quantized(row_fp4, row_sc, row_sg, keepalive=keepalive),
        NVFP4Quantized(col_fp4, col_sc, col_sg, keepalive=keepalive),
    )


def _use_nvfp4_backward_v5_requant() -> bool:
    return _env_flag("FP4_CCE_V4_NVFP4_BACKWARD_V5_REQUANT", False)


def _nvfp4_backward_v5_cols(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    encode_centric: bool,
):
    seed = int(os.environ.get("FP4_CCE_V4_NVFP4_BACKWARD_SR_SEED", "0"))
    rank_subsequence = int(os.environ.get("RANK", "0"))
    x_data_sr = _env_flag(
        "FP4_CCE_V4_NVFP4_BACKWARD_X_DATA_SR", False
    )
    x_scale_sr = _env_flag(
        "FP4_CCE_V4_NVFP4_BACKWARD_X_SCALE_SR", False
    )
    w_data_sr = _env_flag(
        "FP4_CCE_V4_NVFP4_BACKWARD_W_DATA_SR", False
    )
    w_scale_sr = _env_flag(
        "FP4_CCE_V4_NVFP4_BACKWARD_W_SCALE_SR", False
    )
    quantize_x = (
        _quantize_nvfp4_row_and_col_v5_sr
        if x_data_sr or x_scale_sr
        else _quantize_nvfp4_row_and_col_v5
    )
    quantize_w = (
        _quantize_nvfp4_row_and_col_v5_sr
        if w_data_sr or w_scale_sr
        else _quantize_nvfp4_row_and_col_v5
    )
    x_kwargs = (
        {
            "data_stochastic_rounding": x_data_sr,
            "scale_stochastic_rounding": x_scale_sr,
            "rng_seed": seed,
            "rng_subsequence": int(
                os.environ.get(
                    "FP4_CCE_V4_NVFP4_BACKWARD_X_SR_SUBSEQUENCE", "256"
                )
            )
            + rank_subsequence,
        }
        if x_data_sr or x_scale_sr
        else {}
    )
    w_kwargs = (
        {
            "data_stochastic_rounding": w_data_sr,
            "scale_stochastic_rounding": w_scale_sr,
            "rng_seed": seed,
            "rng_subsequence": int(
                os.environ.get(
                    "FP4_CCE_V4_NVFP4_BACKWARD_W_SR_SUBSEQUENCE", "512"
                )
            )
            + rank_subsequence,
        }
        if w_data_sr or w_scale_sr
        else {}
    )
    _x_row_q, x_col_q = quantize_x(
        x,
        encode_centric=encode_centric,
        **x_kwargs,
    )
    _w_row_q, w_col_q = quantize_w(
        weight,
        encode_centric=encode_centric,
        **w_kwargs,
    )
    return x_col_q, w_col_q


def quantize_nvfp4_row_and_col_tk_sr(
    x: torch.Tensor,
    encode_centric: bool = True,
):
    """Quantize BF16 data and its transpose with native FP4 data SR."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    v5 = _get_nvfp4_quant_v5()
    global_scale_target = float(
        os.environ.get("FP4_CCE_V4_NVFP4_P_SCALE_TARGET", "448")
    )
    result = v5.tk_quantize_for_gemm_opt(
        x,
        True,
        bool(encode_centric),
        True,
        _env_flag("FP4_CCE_V4_NVFP4_P_SCALE_SR", False),
        "none",
        False,
        int(os.environ.get("FP4_CCE_V4_NVFP4_P_SR_SEED", "0")),
        int(os.environ.get("FP4_CCE_V4_NVFP4_P_SR_SUBSEQUENCE", "0")),
        global_scale_target,
    )
    row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg = result[:6]
    keepalive = result[6:]
    return (
        NVFP4Quantized(row_fp4, row_sc, row_sg, keepalive=keepalive),
        NVFP4Quantized(col_fp4, col_sc, col_sg, keepalive=keepalive),
    )


def quantize_nvfp4_row_and_col_tk_constant_scale(
    x: torch.Tensor,
    encode_centric: bool = True,
):
    """Quantize unit-bounded BF16 (M, K) and transpose without an amax scan.

    Softmax probabilities and cross-entropy gradients are bounded by one, so
    the v5 quantizer can pin amax=1 without sacrificing representable range.
    """
    if not bool(encode_centric):
        return quantize_nvfp4_row_and_col_tk(x, encode_centric=False)

    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    v5 = _get_nvfp4_quant_v5()
    result = v5.tk_quantize_for_gemm_constant_scale(x, True)
    row_fp4, row_sc, col_fp4, col_sc, sg, _sg2 = result[:6]
    keepalive = result[6:]
    return (
        NVFP4Quantized(row_fp4, row_sc, sg, keepalive=keepalive),
        NVFP4Quantized(col_fp4, col_sc, sg, keepalive=keepalive),
    )


def _rms_norm_bf16_with_inv_rms(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
    with_silu: bool = False,
):
    x_f = x.float()
    inv_rms = torch.rsqrt(x_f.square().mean(dim=-1) + float(epsilon))
    normed = (x_f * inv_rms.unsqueeze(1) * gamma).to(torch.bfloat16)
    if with_silu:
        normed = F.silu(normed.float()).to(torch.bfloat16)
    return normed, inv_rms


def quantize_nvfp4_norm_row_and_col_tk(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    with_silu: bool = False,
    encode_centric: bool = True,
):
    """Fuse RMSNorm(+optional SiLU) with NVFP4 row/col quantization.

    This is the producer-side hook for avoiding a separate CCE input
    quantization pass. The caller is still responsible for preserving the BF16
    producer output if sparse label correction needs it.
    """
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    assert gamma.ndim == 1 and gamma.dtype == torch.bfloat16
    assert x.shape[1] == gamma.shape[0]
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    if (
        not bool(encode_centric)
        or os.environ.get("FP4_CCE_NVFP4_EXACT_NORM_QUANT", "0") != "0"
    ):
        normed, inv_rms = _rms_norm_bf16_with_inv_rms(x, gamma, epsilon, with_silu=with_silu)
        row_q, col_q = quantize_nvfp4_row_and_col_tk(
            normed,
            encode_centric=bool(encode_centric),
        )
        amax = normed.float().abs().amax()
        return row_q, col_q, inv_rms, amax

    if _use_nvfp4_localcta_v4_quant() and not _use_nvfp4_localcta_gemm():
        normed, inv_rms = _rms_norm_bf16_with_inv_rms(x, gamma, epsilon, with_silu=with_silu)
        row_q, col_q = quantize_nvfp4_row_and_col_tk(
            normed,
            encode_centric=bool(encode_centric),
        )
        amax = normed.float().abs().amax()
        return row_q, col_q, inv_rms, amax

    if _use_nvfp4_localcta_v4_quant():
        mod = _get_nvfp4_localcta_v4_quant()
        fn = getattr(mod, "tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt", None)
        if fn is None:
            raise RuntimeError(
                "localCTA v4 quantizer is missing tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt."
            )
        result = fn(
            x,
            gamma,
            float(epsilon),
            True,
            bool(encode_centric),
            False,
            False,
            "none",
            False,
            0,
            0,
            _use_nvfp4_x_four_over_six_mae(),
        )
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms = result[:7]
        keepalive = result[7:]
        amax = torch.empty(0, dtype=torch.float32, device=x.device)
        return (
            NVFP4Quantized(
                row_fp4,
                row_sc,
                row_sg,
                keepalive=keepalive,
                layout="localcta",
            ),
            NVFP4Quantized(
                col_fp4,
                col_sc,
                col_sg,
                keepalive=keepalive,
                layout="localcta",
            ),
            inv_rms,
            amax,
        )

    v5 = _get_nvfp4_quant_v5()
    result = v5.tk_fused_norm_quantize(
        x,
        gamma,
        float(epsilon),
        bool(with_silu),
        True,
    )
    row_fp4, row_sc, col_fp4, col_sc, sg, inv_rms, amax = result[:7]
    keepalive = result[7:]
    return (
        NVFP4Quantized(row_fp4, row_sc, sg, keepalive=keepalive),
        NVFP4Quantized(col_fp4, col_sc, sg, keepalive=keepalive),
        inv_rms,
        amax,
    )


def quantize_nvfp4_norm_row_and_col_with_output_tk(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    with_silu: bool = False,
    encode_centric: bool = True,
):
    """Fuse RMSNorm(+optional SiLU), BF16 output materialization, and NVFP4 row/col quantization."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    assert gamma.ndim == 1 and gamma.dtype == torch.bfloat16
    assert x.shape[1] == gamma.shape[0]
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    if (
        not bool(encode_centric)
        or os.environ.get("FP4_CCE_NVFP4_EXACT_NORM_QUANT", "0") != "0"
    ):
        normed, inv_rms = _rms_norm_bf16_with_inv_rms(x, gamma, epsilon, with_silu=with_silu)
        row_q, col_q = quantize_nvfp4_row_and_col_tk(
            normed,
            encode_centric=bool(encode_centric),
        )
        amax = normed.float().abs().amax()
        return normed, row_q, col_q, inv_rms, amax

    if _use_nvfp4_localcta_v4_quant() and not _use_nvfp4_localcta_gemm():
        normed, inv_rms = _rms_norm_bf16_with_inv_rms(x, gamma, epsilon, with_silu=with_silu)
        row_q, col_q = quantize_nvfp4_row_and_col_tk(
            normed,
            encode_centric=bool(encode_centric),
        )
        amax = normed.float().abs().amax()
        return normed, row_q, col_q, inv_rms, amax

    if _use_nvfp4_localcta_v4_quant():
        mod = _get_nvfp4_localcta_v4_quant()
        fn = getattr(mod, "tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt", None)
        if fn is None:
            raise RuntimeError(
                "localCTA v4 quantizer is missing tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt."
            )
        result = fn(
            x,
            gamma,
            float(epsilon),
            True,
            bool(encode_centric),
            False,
            False,
            "none",
            False,
            0,
            0,
            _use_nvfp4_x_four_over_six_mae(),
        )
        row_fp4, row_sc, col_fp4, col_sc, row_sg, col_sg, inv_rms = result[:7]
        keepalive = result[7:]
        normed = (x.float() * inv_rms.unsqueeze(1) * gamma).to(torch.bfloat16)
        if with_silu:
            normed = F.silu(normed.float()).to(torch.bfloat16)
        amax = torch.empty(0, dtype=torch.float32, device=x.device)
        return (
            normed,
            NVFP4Quantized(
                row_fp4,
                row_sc,
                row_sg,
                keepalive=keepalive,
                layout="localcta",
            ),
            NVFP4Quantized(
                col_fp4,
                col_sc,
                col_sg,
                keepalive=keepalive,
                layout="localcta",
            ),
            inv_rms,
            amax,
        )

    v5 = _get_nvfp4_quant_v5()
    fused_with_output = getattr(v5, "tk_fused_norm_quantize_with_output", None)
    if fused_with_output is not None:
        result = fused_with_output(
            x,
            gamma,
            float(epsilon),
            bool(with_silu),
            True,
        )
        normed, row_fp4, row_sc, col_fp4, col_sc, sg, inv_rms, amax = result[:8]
        keepalive = result[8:]
    else:
        result = v5.tk_fused_norm_quantize(
            x,
            gamma,
            float(epsilon),
            bool(with_silu),
            True,
        )
        row_fp4, row_sc, col_fp4, col_sc, sg, inv_rms, amax = result[:7]
        keepalive = result[7:]
        normed = (x.float() * inv_rms.unsqueeze(1) * gamma).to(torch.bfloat16)
        if with_silu:
            normed = F.silu(normed.float()).to(torch.bfloat16)
    return (
        normed,
        NVFP4Quantized(row_fp4, row_sc, sg, keepalive=keepalive),
        NVFP4Quantized(col_fp4, col_sc, sg, keepalive=keepalive),
        inv_rms,
        amax,
    )


def _use_nvfp4_p_constant_scale() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE", "0") != "0"


def _use_nvfp4_p_data_sr() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_P_DATA_SR", "0") != "0"


def _use_nvfp4_p_target_split() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_P_TARGET_SPLIT", "0") != "0"


def _use_nvfp4_exact_target_logit() -> bool:
    return os.environ.get("FP4_CCE_V4_EXACT_TARGET_LOGIT", "0") != "0"


def _use_nvfp4_exact_selected_logits() -> bool:
    return os.environ.get("FP4_CCE_V4_EXACT_TARGET_TOPK_LOGITS", "0") != "0"


def _use_sparse_repair_overlap() -> bool:
    return os.environ.get("FP4_CCE_V4_SPARSE_REPAIR_OVERLAP", "0") != "0"


def _use_mx_compact_dw_repair() -> bool:
    value = os.environ.get("FP4_CCE_V4_MX_COMPACT_DW_REPAIR")
    if value is None:
        value = os.environ.get("FP4_CCE_V4_MXFP4_COMPACT_DW_REPAIR", "1")
    return value != "0"


def _use_mxfp4_g_atbt_dw() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_G_ATBT_DW", "0") != "0"


def _use_mx_backward_gemm_overlap() -> bool:
    return os.environ.get("FP4_CCE_V4_MX_BACKWARD_GEMM_OVERLAP", "0") != "0"


def _mx_backward_gemm_stream(device: torch.device):
    if torch.cuda.is_current_stream_capturing():
        return None
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _mx_backward_gemm_streams.get(index)
    if stream is None:
        with torch.cuda.device(index):
            stream = torch.cuda.Stream(device=index)
        _mx_backward_gemm_streams[index] = stream
    return stream


def _sparse_dC_preparation_stream(device: torch.device):
    if torch.cuda.is_current_stream_capturing():
        return None
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _sparse_dC_preparation_streams.get(index)
    if stream is None:
        with torch.cuda.device(index):
            stream = torch.cuda.Stream(device=index)
        _sparse_dC_preparation_streams[index] = stream
    return stream


def _sparse_repair_stream(device: torch.device):
    if torch.cuda.is_current_stream_capturing():
        return None
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    stream = _sparse_repair_streams.get(index)
    if stream is None:
        with torch.cuda.device(index):
            stream = torch.cuda.Stream(device=index)
        _sparse_repair_streams[index] = stream
    return stream


def _retain_sparse_dC_inputs_until_complete(
    device: torch.device,
    tensors: tuple[torch.Tensor, ...],
    stream=None,
) -> None:
    """Keep custom-kernel inputs alive until their CUDA work completes."""
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    queue = _sparse_dC_inflight.setdefault(index, [])
    queue[:] = [item for item in queue if not item[0].query()]
    completion = torch.cuda.Event()
    completion.record(
        torch.cuda.current_stream(device) if stream is None else stream
    )
    queue.append((completion, tensors))


def _launch_overlapped_topk_dE_repair(
    dE,
    weight,
    targets,
    target_probs,
    topk_indices,
    topk_probs,
    scale,
    ignore_index,
):
    if not _use_sparse_repair_overlap():
        return None
    repair_stream = _sparse_repair_stream(dE.device)
    if repair_stream is None:
        return None
    current_stream = torch.cuda.current_stream(dE.device)
    repair_stream.wait_stream(current_stream)
    with torch.cuda.stream(repair_stream):
        sparse_correct_target_topk_dE(
            dE,
            weight,
            targets,
            target_probs,
            topk_indices,
            topk_probs,
            scale,
            ignore_index,
        )
    dE.record_stream(repair_stream)
    return repair_stream


def _finish_overlapped_topk_repair(
    repair_stream,
    dC,
    x,
    targets,
    target_probs,
    topk_indices,
    topk_probs,
    scale,
    ignore_index,
    prepared_dC=None,
):
    current_stream = torch.cuda.current_stream(dC.device)
    if prepared_dC is not None:
        current_stream.wait_stream(repair_stream)
    if prepared_dC is None:
        sparse_correct_target_topk_dC(
            dC,
            x,
            targets,
            target_probs,
            topk_indices,
            topk_probs,
            scale,
            ignore_index,
        )
    else:
        for tensor in prepared_dC:
            tensor.record_stream(current_stream)
        scaled_coefficients = sparse_correct_target_topk_dC_prepared(
            dC,
            x,
            prepared_dC,
            scale,
        )
        _retain_sparse_dC_inputs_until_complete(
            dC.device,
            (*prepared_dC, scaled_coefficients),
        )
    if prepared_dC is None:
        current_stream.wait_stream(repair_stream)


def _launch_compact_topk_dC_repair(
    preparation_stream,
    x,
    targets,
    target_probs,
    topk_indices,
    topk_probs,
    vocab_size,
    ignore_index,
):
    """Enqueue compact repair accumulation alongside the backward GEMMs."""
    if preparation_stream is None:
        return None
    with torch.cuda.stream(preparation_stream):
        compact = compact_target_topk_dC(
            x,
            targets,
            target_probs,
            topk_indices,
            topk_probs,
            vocab_size,
            ignore_index,
        )
    for tensor in compact:
        tensor.record_stream(preparation_stream)
    return preparation_stream, compact


def _finish_compact_topk_dC_repair(dC, compact_repair, scale):
    preparation_stream, compact = compact_repair
    current_stream = torch.cuda.current_stream(dC.device)
    current_stream.wait_stream(preparation_stream)
    for tensor in compact:
        tensor.record_stream(current_stream)
    add_compact_target_topk_dC(dC, *compact, scale)


def _use_nvfp4_bf16_logits() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_BF16_LOGITS", "0") != "0"


def _use_analysis_bf16_logits_with_mxfp8_backward() -> bool:
    return _env_flag(
        "FP4_CCE_ANALYSIS_BF16_LOGITS_WITH_MXFP8_BACKWARD", False
    )


def _use_nvfp4_p_top1_split() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_P_TOP1_SPLIT", "0") != "0"


def _nvfp4_p_topk_split() -> int:
    raw = os.environ.get("FP4_CCE_V4_NVFP4_P_TOPK_SPLIT")
    topk = int(raw) if raw is not None else int(_use_nvfp4_p_top1_split())
    if topk not in {0, 1, 4}:
        raise ValueError("FP4_CCE_V4_NVFP4_P_TOPK_SPLIT must be 0, 1, or 4")
    return topk


def _use_nvfp4_xw_constant_scale() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_XW_CONSTANT_SCALE", "0") != "0"


def _use_nvfp4_fused_staged_p_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_FUSED_STAGED_P_CACHE", "0") != "0"


def _use_nvfp4_bf16_p_dweight() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_BF16_P_DWEIGHT", "0") != "0"


def _use_nvfp4_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_G_CACHE", "1") != "0"


def _use_nvfp4_g_target_split() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_G_TARGET_SPLIT", "0") != "0"


def _nvfp4_g_topk_split() -> int:
    topk = int(os.environ.get("FP4_CCE_V4_NVFP4_G_TOPK_SPLIT", "0"))
    if topk not in {0, 1, 2, 4, 6, 8, 12, 16}:
        raise ValueError(
            "FP4_CCE_V4_NVFP4_G_TOPK_SPLIT must be 0, 1, 2, 4, 6, 8, 12, or 16"
        )
    return topk


def _use_nvfp4_g_constant_scale() -> bool:
    # G = softmax(logits) - onehot has theoretical absmax 1, unlike x/weight.
    # The constant-scale shortcut removes the quantizer amax pass and measured
    # close to dynamic G numerics on the 1.2B final-layer CCE shape. Set the env
    # var to 0 to force the older dynamic path for ablations.
    return os.environ.get("FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE", "1") != "0"


def _nvfp4_g_scale_max() -> float:
    scale_max = float(os.environ.get("FP4_CCE_V4_NVFP4_G_SCALE_MAX", "448"))
    if not math.isfinite(scale_max) or scale_max <= 0.0:
        raise ValueError("FP4_CCE_V4_NVFP4_G_SCALE_MAX must be finite and positive")
    return scale_max


def _use_nvfp4_g_chunk_scale() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_G_CHUNK_SCALE", "0") != "0"


def _use_mxfp8_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP8_G_CACHE", "0") != "0"


def _use_mxfp4_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_G_CACHE", "0") != "0"


def _use_mxfp8_row_nvfp4_col_g_cache() -> bool:
    return (
        os.environ.get(
            "FP4_CCE_V4_MXFP8_ROW_NVFP4_COL_G_CACHE", "0"
        )
        != "0"
    )


def _use_mixed_dw_mxfp8_cols() -> bool:
    """Keep the deployed mixed head but use MXFP8 columns for dWeight."""
    return _env_flag("FP4_CCE_V4_MIXED_DW_MXFP8_COLS", False)


def _use_lowp_logits_bf16_dhidden() -> bool:
    """Use corrected low-precision probabilities with the original BF16 W."""
    return _env_flag("FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN", False)


def _use_lowp_logits_bf16_dhidden_inplace_g() -> bool:
    """Reuse dead BF16 logits storage for the accurate repaired G operand."""
    return _env_flag(
        "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN_INPLACE_G", False
    )


def _use_lowp_logits_bf16_dweight() -> bool:
    """Reduce the repaired low-precision-logit G against BF16 hidden states."""
    return _env_flag("FP4_CCE_V4_LOWP_LOGITS_BF16_DWEIGHT", False)


def _use_lowp_logits_bf16_both_inplace_g() -> bool:
    """Select the cache-elision-safe lowp-forward/BF16-both-gradient mode."""
    return bool(
        _use_lowp_logits_bf16_dhidden()
        and _use_lowp_logits_bf16_dhidden_inplace_g()
        and _use_lowp_logits_bf16_dweight()
    )


def _mxfp8_logit_temperature() -> float:
    temperature = float(
        os.environ.get(
            "FP4_CCE_V4_LOGIT_TEMPERATURE",
            os.environ.get("FP4_CCE_V4_MXFP8_LOGIT_TEMPERATURE", "1"),
        )
    )
    if not math.isfinite(temperature) or not 0.5 <= temperature <= 2.0:
        raise ValueError(
            "FP4_CCE_V4_LOGIT_TEMPERATURE must be in [0.5, 2]"
        )
    return temperature


def _validate_lowp_logits_bf16_dhidden(
    *,
    use_mxfp8_forward: bool,
    use_mixed_g_cache: bool,
    use_mixed_dw_mxfp8_cols: bool,
    g_target_split: bool,
    g_topk_split: int,
) -> None:
    """Fail closed outside the measured mixed-MXFP8 output-head path."""
    if not (
        use_mxfp8_forward
        and use_mixed_g_cache
        and use_mixed_dw_mxfp8_cols
        and g_target_split
    ):
        raise RuntimeError(
            "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1 requires MXFP8 forward, "
            "the mixed MXFP8-row cache, and MXFP8 dWeight columns"
        )
    if not (
        _use_nvfp4_fused_g_cache()
        and _nvfp4_fused_g_cache_impl() == "tiled"
        and _env_flag("FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW", False)
    ):
        raise RuntimeError(
            "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1 requires the tiled fused "
            "MXFP8 softmax-row producer"
        )
    exact_topk = int(os.environ.get("FP4_CCE_V4_EXACT_SELECTED_TOPK", "0"))
    if not _use_nvfp4_exact_selected_logits() or (
        g_topk_split,
        exact_topk,
    ) != (16, 16):
        raise RuntimeError(
            "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1 requires exact target/top-16 "
            "logits and repair"
        )
    if _use_nvfp4_exact_target_logit():
        raise RuntimeError(
            "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1 rejects the legacy exact "
            "target-logit replacement"
        )
    if _env_flag("FP4_CCE_V4_MXFP8_FP8_LOGITS", False) or _env_flag(
        "FP4_CCE_V4_MXFP8_CENTERED_FP8_LOGITS", False
    ):
        raise RuntimeError(
            "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1 requires materialized BF16 "
            "MXFP8 logits"
        )
    if _use_sparse_repair_overlap() or _use_mx_backward_gemm_overlap():
        raise RuntimeError(
            "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1 is validated only with "
            "backward GEMM and sparse-repair overlap disabled"
        )


def _mxfp8_g_quant_max() -> float:
    quant_max = float(os.environ.get("FP4_CCE_V4_MXFP8_G_QUANT_MAX", "448"))
    if not math.isfinite(quant_max) or quant_max <= 0.0:
        raise ValueError("FP4_CCE_V4_MXFP8_G_QUANT_MAX must be finite and positive")
    return quant_max


def _use_nvfp4_fused_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_FUSED_G_CACHE", "0") != "0"


def _use_nvfp4_staged_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_STAGED_G_CACHE", "0") != "0"


def _nvfp4_fused_g_cache_impl() -> str:
    return os.environ.get("FP4_CCE_V4_NVFP4_FUSED_G_CACHE_IMPL", "direct").strip().lower()


def _nvfp4_target_split_g_cache(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    *,
    encode_centric: bool,
):
    topk_split = _nvfp4_g_topk_split()
    exact_selected_logits = _use_nvfp4_exact_selected_logits()
    if exact_selected_logits and _use_nvfp4_exact_target_logit():
        raise RuntimeError(
            "exact target/top-k repair replaces FP4_CCE_V4_EXACT_TARGET_LOGIT"
        )
    if (
        _use_nvfp4_fused_g_cache()
        and encode_centric
        and _nvfp4_fused_g_cache_impl() == "tiled"
    ):
        return nvfp4_tiled_g_cache_target_split(
            logits,
            targets,
            valid,
            vocab_size,
            topk_split,
            x=x,
            weight=weight,
            exact_selected_logits=exact_selected_logits,
            constant_scale=_use_nvfp4_g_constant_scale(),
            global_scale_max=_nvfp4_g_scale_max(),
            chunk_scale=_use_nvfp4_g_chunk_scale(),
        )

    if exact_selected_logits:
        raise RuntimeError(
            "exact target/top-k repair requires the fused tiled G-cache producer"
        )

    if topk_split == 6:
        raise RuntimeError(
            "NVFP4 G top-6 split requires the fused tiled G-cache producer"
        )
    if topk_split == 4:
        (
            loss,
            grad_probs,
            target_probs,
            topk_probs,
            topk_indices,
        ) = direct_loss_and_probs_target_top4_split(
            logits, targets, valid, vocab_size
        )
    elif topk_split == 2:
        (
            loss,
            grad_probs,
            target_probs,
            topk_probs,
            topk_indices,
        ) = direct_loss_and_probs_target_top2_split(
            logits, targets, valid, vocab_size
        )
    elif topk_split == 1:
        (
            loss,
            grad_probs,
            target_probs,
            topk_probs,
            topk_indices,
        ) = direct_loss_and_probs_target_top1_split(
            logits, targets, valid, vocab_size
        )
    else:
        loss, grad_probs, target_probs = direct_loss_and_probs_target_split(
            logits, targets, valid, vocab_size
        )
        topk_probs = None
        topk_indices = None
    g_row_q, g_col_q = _select_nvfp4_p_quantizer()(
        grad_probs, encode_centric=encode_centric
    )
    return (
        loss,
        g_row_q.fp4,
        g_row_q.sc,
        g_row_q.sg,
        g_col_q.fp4,
        g_col_q.sc,
        g_col_q.sg,
        target_probs,
        topk_indices,
        topk_probs,
    )


def _mxfp8_target_split_g_cache(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
):
    topk_split = _nvfp4_g_topk_split()
    exact_selected_logits = _use_nvfp4_exact_selected_logits()
    if not _use_nvfp4_g_target_split():
        raise RuntimeError("MXFP8 G-cache requires target splitting")
    if exact_selected_logits and _use_nvfp4_exact_target_logit():
        raise RuntimeError(
            "exact target/top-k repair replaces FP4_CCE_V4_EXACT_TARGET_LOGIT"
        )
    if exact_selected_logits and topk_split == 0:
        raise RuntimeError(
            "exact target/top-k repair requires FP4_CCE_V4_NVFP4_G_TOPK_SPLIT"
        )
    if not _use_nvfp4_fused_g_cache() or _nvfp4_fused_g_cache_impl() != "tiled":
        raise RuntimeError("MXFP8 G-cache requires the fused tiled producer")
    result = mxfp8_tiled_g_cache_target_split(
        logits,
        targets,
        valid,
        vocab_size,
        topk_split,
        x=x,
        weight=weight,
        exact_selected_logits=exact_selected_logits,
        quant_max=_mxfp8_g_quant_max(),
    )
    preparation_stream = None
    if topk_split and _use_sparse_repair_overlap() and _use_mx_compact_dw_repair():
        preparation_stream = _sparse_dC_preparation_stream(logits.device)
        if preparation_stream is not None:
            preparation_stream.wait_stream(
                torch.cuda.current_stream(logits.device)
            )
    return (*result, preparation_stream)


def _mxfp8_row_nvfp4_col_target_split_g_cache(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    vocab_size: int,
    *,
    return_lse: bool = False,
):
    topk_split = _nvfp4_g_topk_split()
    exact_selected_logits = _use_nvfp4_exact_selected_logits()
    if not _use_nvfp4_g_target_split():
        raise RuntimeError("mixed MXFP8/NVFP4 G-cache requires target splitting")
    if exact_selected_logits and _use_nvfp4_exact_target_logit():
        raise RuntimeError(
            "exact target/top-k repair replaces FP4_CCE_V4_EXACT_TARGET_LOGIT"
        )
    if exact_selected_logits and topk_split == 0:
        raise RuntimeError(
            "exact target/top-k repair requires FP4_CCE_V4_NVFP4_G_TOPK_SPLIT"
        )
    if not _use_nvfp4_fused_g_cache() or _nvfp4_fused_g_cache_impl() != "tiled":
        raise RuntimeError("mixed MXFP8/NVFP4 G-cache requires the tiled producer")
    fused_row = (
        os.environ.get("FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW", "0") != "0"
    )
    row_normalization = None
    corrected_lse = None
    if fused_row:
        if topk_split not in (0, 8, 12, 16):
            raise RuntimeError(
                "fused MXFP8 softmax-row production requires top-k 0, 8, 12, or 16"
            )
        (
            loss,
            corrected_lse,
            target_probs,
            topk_probs,
            topk_indices,
            row_fp8,
            row_sc,
            _row_sg,
            row_normalization,
        ) = direct_loss_lse_target_topk_split_exact_logits_mxfp8_row(
            logits,
            x,
            weight,
            targets,
            valid,
            vocab_size,
            topk_split,
            _mxfp8_g_quant_max(),
            exact_selected_logits,
        )
        if topk_split == 0:
            topk_probs = None
            topk_indices = None
        if _use_lowp_logits_bf16_both_inplace_g():
            empty_col = _empty_mxfp8_quantized(logits.device)
            col_fp4, col_sc = empty_col.fp8, empty_col.sc
            col_sg = torch.empty(
                0, dtype=torch.float32, device=logits.device
            )
        elif _use_mixed_dw_mxfp8_cols():
            col_fp4, col_sc = mxfp8_col_requant_from_mxfp8_row(
                row_fp8,
                row_sc,
                row_normalization,
                _mxfp8_g_quant_max(),
            )
            col_sg = torch.empty(
                0, dtype=torch.float32, device=logits.device
            )
        else:
            col_fp4, col_sc, col_sg = nvfp4_col_requant_from_mxfp8_row(
                row_fp8,
                row_sc,
                row_normalization,
                _nvfp4_g_scale_max(),
            )
        result = (
            loss,
            row_fp8,
            row_sc,
            col_fp4,
            col_sc,
            col_sg,
            target_probs,
            topk_indices,
            topk_probs,
        )
    else:
        if _use_mixed_dw_mxfp8_cols():
            raise RuntimeError(
                "MXFP8 dWeight columns require the fused MXFP8 softmax-row "
                "producer"
            )
        result = mxfp8_row_nvfp4_col_tiled_g_cache_target_split(
            logits,
            targets,
            valid,
            vocab_size,
            topk_split,
            x=x,
            weight=weight,
            exact_selected_logits=exact_selected_logits,
            mxfp8_quant_max=_mxfp8_g_quant_max(),
            nvfp4_global_scale_max=_nvfp4_g_scale_max(),
        )
    preparation_stream = None
    if topk_split and _use_sparse_repair_overlap() and _use_mx_compact_dw_repair():
        preparation_stream = _sparse_dC_preparation_stream(logits.device)
        if preparation_stream is not None:
            preparation_stream.wait_stream(
                torch.cuda.current_stream(logits.device)
            )
    result = (*result, preparation_stream)
    if row_normalization is not None:
        result = (*result, row_normalization)
    if return_lse:
        if corrected_lse is None:
            raise RuntimeError(
                "corrected LSE is available only from the fused MXFP8 row producer"
            )
        result = (*result, corrected_lse)
    return result


def _mxfp4_target_split_g_cache(
    logits: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
    vocab_size: int,
    logit_centers: torch.Tensor | None = None,
):
    topk_split = _nvfp4_g_topk_split()
    exact_selected_logits = _use_nvfp4_exact_selected_logits()
    if not _use_nvfp4_g_target_split():
        raise RuntimeError("MXFP4 G-cache requires target splitting")
    if exact_selected_logits and _use_nvfp4_exact_target_logit():
        raise RuntimeError(
            "exact target/top-k repair replaces FP4_CCE_V4_EXACT_TARGET_LOGIT"
        )
    if not _use_nvfp4_fused_g_cache() or _nvfp4_fused_g_cache_impl() != "tiled":
        raise RuntimeError("MXFP4 G-cache requires the fused tiled producer")

    fused_row = (
        os.environ.get("FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW", "0") != "0"
    )
    if logits.dtype == torch.float8_e4m3fn and not fused_row:
        raise RuntimeError(
            "direct E4M3 logits require "
            "FP4_CCE_V4_NVFP4_G_FUSED_SOFTMAX_ROW=1"
        )
    row_normalization = None
    if fused_row:
        if topk_split not in (0, 8, 12, 16):
            raise RuntimeError(
                "fused MXFP4 softmax-row production requires top-k 0, 8, 12, or 16"
            )
        if os.environ.get("FP4_CCE_V4_MXFP4_G_DATA_SR", "0") != "0":
            raise RuntimeError(
                "fused MXFP4 softmax-row production requires orientation-specific "
                "SR flags; full row+column SR is not supported"
            )
        if (
            os.environ.get("FP4_CCE_V4_MXFP4_G_COL_DATA_SR", "0") != "0"
            and os.environ.get("FP4_CCE_V4_MXFP4_G_COL_ZERO_SR", "0") != "0"
        ):
            raise RuntimeError(
                "full and zero-only MXFP4 column stochastic rounding are "
                "mutually exclusive"
            )
        if os.environ.get("FP4_CCE_V4_MXFP4_G_LOG_CODE", "0") != "0":
            raise RuntimeError(
                "fused MXFP4 softmax-row production requires linear E2M1 coding"
            )
        scale_floor_ratio = float(
            os.environ.get("FP4_CCE_V4_MXFP4_G_SCALE_FLOOR_RATIO", "1.125")
        )
        (
            loss,
            _lse,
            target_probs,
            topk_probs,
            topk_indices,
            row_fp4,
            row_sc,
            _row_sg,
            row_normalization,
        ) = (
            direct_loss_lse_target_topk_split_exact_logits_mxfp4_row_centered(
                logits,
                logit_centers,
                x,
                weight,
                targets,
                valid,
                vocab_size,
                topk_split,
                scale_floor_ratio,
                exact_selected_logits,
            )
            if logit_centers is not None
            else direct_loss_lse_target_topk_split_exact_logits_mxfp4_row(
                logits,
                x,
                weight,
                targets,
                valid,
                vocab_size,
                topk_split,
                scale_floor_ratio,
                exact_selected_logits,
            )
        )
        if topk_split == 0:
            topk_probs = None
            topk_indices = None
        if _use_mxfp4_g_atbt_dw():
            col_fp4 = torch.empty(
                0, dtype=torch.float4_e2m1fn_x2, device=row_fp4.device
            )
            col_sc = torch.empty(0, dtype=torch.uint8, device=row_fp4.device)
        else:
            col_fp4, col_sc = mxfp4_col_requant_from_row(
                row_fp4, row_sc, row_normalization
            )
    else:
        (
            loss,
            lse,
            target_probs,
            topk_probs,
            topk_indices,
        ) = (
            direct_loss_lse_target_topk_split_exact_logits(
                logits,
                x,
                weight,
                targets,
                valid,
                vocab_size,
                topk_split,
            )
            if exact_selected_logits
            else direct_loss_lse_target_topk_split(
                logits,
                targets,
                valid,
                vocab_size,
                topk_split,
            )
        )
        if topk_split == 0:
            topk_probs = None
            topk_indices = None

        row_fp4, row_sc, col_fp4, col_sc = mxfp4_softmax_tail_quant_row_col(
            logits,
            lse,
            valid,
            vocab_size,
        )
    prepared_dC = None
    preparation_stream = None
    prepare_sparse_dC = (
        os.environ.get("FP4_CCE_V4_PREPARE_SPARSE_DC", "1") != "0"
    )
    compact_output = _use_mx_compact_dw_repair()
    if (
        topk_split
        and _use_sparse_repair_overlap()
        and (compact_output or prepare_sparse_dC)
    ):
        preparation_stream = (
            _sparse_dC_preparation_stream(logits.device)
            if compact_output
            else _sparse_repair_stream(logits.device)
        )
        if preparation_stream is not None:
            current_stream = torch.cuda.current_stream(logits.device)
            preparation_stream.wait_stream(current_stream)
            if not compact_output:
                with torch.cuda.stream(preparation_stream):
                    prepared_dC = prepare_sparse_correct_target_topk_dC(
                        targets,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        ignore_index,
                        valid,
                    )
                preparation_inputs = (
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    valid,
                )
                for tensor in preparation_inputs:
                    tensor.record_stream(preparation_stream)
                _retain_sparse_dC_inputs_until_complete(
                    logits.device,
                    preparation_inputs,
                    stream=preparation_stream,
                )
                for tensor in prepared_dC:
                    tensor.record_stream(preparation_stream)
    result = (
        loss,
        row_fp4,
        row_sc,
        col_fp4,
        col_sc,
        target_probs,
        topk_indices,
        topk_probs,
        prepared_dC,
        preparation_stream,
    )
    if row_normalization is not None:
        return (*result, row_normalization)
    return result


def _use_nvfp4_direct_g_producer() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_DIRECT_G_PRODUCER", "1") != "0"


def _nvfp4_bf16_p_dweight_chunk() -> int:
    return max(int(os.environ.get("FP4_CCE_V4_BF16_P_DWEIGHT_CHUNK", "4096")), 256)


def _use_nvfp4_group_input_quant() -> bool:
    return os.environ.get("FP4_CCE_V4_NVFP4_GROUP_INPUT_QUANT", "0") != "0"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _select_nvfp4_gemm_role_config(role: str, stable: int, tuned: int) -> int:
    override = _env_int(f"FP4_CCE_V4_NVFP4_GEMM_CONFIG_{role}")
    if override is not None:
        return override
    if _env_flag("FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED", False):
        return tuned
    return stable


def _select_nvfp4_gemm_config(M: int, N: int, K: int) -> int | None:
    forced = os.environ.get("FP4_CCE_V4_NVFP4_GEMM_CONFIG_ID")
    if forced is not None and forced.strip():
        return int(forced)
    if os.environ.get("FP4_CCE_V4_NVFP4_GEMM_CONFIG_AUTO", "1") == "0":
        return None

    # Forward logits: [tokens, 2048] x [vocab, 2048]^T.
    if K == 2048 and N >= 65536:
        return _select_nvfp4_gemm_role_config("FWD", stable=15, tuned=12)

    # dC: [vocab, tokens] x [2048, tokens]^T.
    if N == 2048 and M >= 131072 and K <= 65536:
        return _select_nvfp4_gemm_role_config("DC", stable=13, tuned=0)

    # dE: [tokens, vocab] x [2048, vocab]^T.
    if N == 2048 and K >= 65536:
        return _select_nvfp4_gemm_role_config("DE", stable=13, tuned=1)

    # Smaller dC-style shapes from chunked experiments.
    if N == 2048 and M >= 131072:
        return 5
    if N == 2048 and M >= 65536 and K >= 8192:
        return 15
    if N == 2048 and M >= 65536 and K >= 4096:
        return 14

    return None


def _select_nvfp4_xw_quantizer(role: str | None = None):
    if role not in {None, "X", "W"}:
        raise ValueError(f"NVFP4 head quantization role must be X or W, got {role!r}")
    if _use_nvfp4_xw_constant_scale():
        four_over_six_mae = (
            role == "X" and _use_nvfp4_x_four_over_six_mae()
        ) or (
            role == "W" and _use_nvfp4_w_four_over_six_mae()
        )
        if four_over_six_mae:
            raise RuntimeError(
                "localCTA FourOverSix MAE requires dynamic X/W scaling; disable "
                "FP4_CCE_V4_NVFP4_XW_CONSTANT_SCALE"
            )
        return quantize_nvfp4_row_and_col_tk_constant_scale
    if role is None:
        return quantize_nvfp4_xw_row_and_col_tk

    def quantize_role(x: torch.Tensor, encode_centric: bool = True):
        return quantize_nvfp4_xw_row_and_col_tk(
            x,
            encode_centric=encode_centric,
            role=role,
        )

    return quantize_role


def _select_nvfp4_g_quantizer():
    if _use_nvfp4_g_constant_scale():
        return quantize_nvfp4_row_and_col_tk_constant_scale
    if _use_nvfp4_backward_v5_requant():
        return _quantize_nvfp4_row_and_col_v5
    return quantize_nvfp4_row_and_col_tk


def _select_nvfp4_p_quantizer():
    if _use_nvfp4_p_constant_scale():
        if _use_nvfp4_p_data_sr():
            raise RuntimeError(
                "NVFP4 P data SR requires dynamic P scaling; set "
                "FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE=0"
            )
        return quantize_nvfp4_row_and_col_tk_constant_scale
    if _use_nvfp4_p_data_sr():
        return quantize_nvfp4_row_and_col_tk_sr
    if _use_nvfp4_backward_v5_requant():
        return _quantize_nvfp4_row_and_col_v5
    return quantize_nvfp4_row_and_col_tk


def _scaled_g_cache_operand(
    fp4: torch.Tensor,
    sc: torch.Tensor,
    sg: torch.Tensor,
    scale: torch.Tensor,
    *,
    localcta_constant_layout: bool,
) -> NVFP4Quantized:
    scaled_sg = sg * scale
    if not localcta_constant_layout:
        return NVFP4Quantized(fp4, sc, scaled_sg)

    rows = int(fp4.shape[0])
    if rows % 256:
        raise RuntimeError(
            "constant-scale G can use the localCTA backward layout only when "
            f"its row dimension is divisible by 256, got {rows}"
        )
    outer_rows = rows // 256
    flat_sg = scaled_sg.reshape(-1)
    if flat_sg.numel() == 1:
        localcta_sg = flat_sg.expand(outer_rows).reshape(outer_rows, 1).contiguous()
    elif flat_sg.numel() == outer_rows:
        localcta_sg = flat_sg.reshape(outer_rows, 1).contiguous()
    else:
        raise RuntimeError(
            "constant-scale G outer-scale count is incompatible with localCTA: "
            f"rows={rows}, sg={tuple(sg.shape)}"
        )
    return NVFP4Quantized(
        fp4,
        sc,
        localcta_sg,
        layout="localcta",
    )


def quantize_both_nvfp4_tk(x: torch.Tensor, w: torch.Tensor,
                           keep_bf16: bool = True):
    """Quantize both x(M,K) and w(V,K) in a single grouped kernel launch.

    Returns (x_q: NVFP4Quantized, w_q: NVFP4Quantized).
    """
    assert x.ndim == 2 and w.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    V = w.shape[0]
    assert w.shape[1] == K
    assert K % 128 == 0 and M % 128 == 0 and V % 128 == 0

    # Stack vertically and quantize in one launch
    stacked = torch.cat([x, w], dim=0)  # (M+V, K)
    v5 = _get_nvfp4_quant_v5()
    result = v5.tk_group_quantize_for_gemm(stacked, [M, V])

    # Result: (row_fp4, row_sc, fwd_b_sg, col_fp4_list, col_sc_list, dgrad_b_sg, sg_cat, mega_buf)
    row_fp4, row_sc, _fwd_b_sg, _col_fp4_list, _col_sc_list, _dgrad_b_sg, sg_cat, _mega_buf = result

    # Split FP4 and scales
    x_fp4 = row_fp4[:M].contiguous()
    w_fp4 = row_fp4[M:].contiguous()

    # Scales are already per-split from group quantize, but returned concatenated
    # sc shape: [total_rows/128, K_tiles, 512] — need to split at M/128
    sc_tiles_x = M // 128
    x_sc = row_sc[:sc_tiles_x].contiguous()
    w_sc = row_sc[sc_tiles_x:].contiguous()

    # sg is per-split: [2] tensor
    x_sg = sg_cat[0:1].contiguous()
    w_sg = sg_cat[1:2].contiguous()

    x_q = NVFP4Quantized(x_fp4, x_sc, x_sg, bf16=x if keep_bf16 else None)
    w_q = NVFP4Quantized(w_fp4, w_sc, w_sg, bf16=w if keep_bf16 else None)
    return x_q, w_q


def quantize_both_nvfp4_row_and_col_tk(
    x: torch.Tensor,
    w: torch.Tensor,
    encode_centric: bool = True,
):
    """Grouped NVFP4 row/col quantization for P-cache forward inputs.

    This saves one quantizer launch/barrier versus quantizing x and w
    independently, but it first materializes a stacked BF16 input, so keep it
    behind a benchmark gate.
    """
    if not bool(encode_centric) or _use_nvfp4_localcta_v4_quant():
        quantize_x = _select_nvfp4_xw_quantizer("X")
        quantize_w = _select_nvfp4_xw_quantizer("W")
        x_row_q, x_col_q = quantize_x(
            x,
            encode_centric=bool(encode_centric),
        )
        w_row_q, w_col_q = quantize_w(
            w,
            encode_centric=bool(encode_centric),
        )
        return x_row_q, x_col_q, w_row_q, w_col_q

    assert x.ndim == 2 and w.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    V = w.shape[0]
    assert w.shape[1] == K
    assert K % 128 == 0 and M % 128 == 0 and V % 128 == 0

    stacked = torch.cat([x, w], dim=0)
    v5 = _get_nvfp4_quant_v5()
    row_fp4, row_sc, _fwd_b_sg, col_fp4_list, col_sc_list, _dgrad_b_sg, sg_cat, _mega_buf = (
        v5.tk_group_quantize_for_gemm(stacked, [M, V])
    )

    x_sc_tiles = M // 128
    x_row_q = NVFP4Quantized(row_fp4[:M], row_sc[:x_sc_tiles], sg_cat[0:1])
    w_row_q = NVFP4Quantized(row_fp4[M:], row_sc[x_sc_tiles:], sg_cat[1:2])
    x_col_q = NVFP4Quantized(col_fp4_list[0], col_sc_list[0], sg_cat[0:1])
    w_col_q = NVFP4Quantized(col_fp4_list[1], col_sc_list[1], sg_cat[1:2])
    return x_row_q, x_col_q, w_row_q, w_col_q


def tk_nvfp4_gemm(A_q: NVFP4Quantized, B_q: NVFP4Quantized,
                  out: torch.Tensor = None) -> torch.Tensor:
    """D(M, N) = A @ B^T using TK NVFP4 GEMM."""
    M = A_q.fp4.shape[0]
    N = B_q.fp4.shape[0]
    K = A_q.fp4.shape[1] * 2
    use_empty = os.environ.get(
        "FP4_CCE_V4_NVFP4_EMPTY_GEMM_OUTPUT",
        os.environ.get("FP4_CCE_V4_EMPTY_GEMM_OUTPUT", "1"),
    ) == "1"
    if out is None:
        if use_empty:
            out = torch.empty(M, N, dtype=torch.bfloat16, device=A_q.fp4.device)
        else:
            out = torch.zeros(M, N, dtype=torch.bfloat16, device=A_q.fp4.device)
    else:
        if not use_empty:
            out.zero_()
    if _use_nvfp4_localcta_gemm(A_q, B_q):
        local_gemm = _get_tk_nvfp4_localcta_gemm_v3()
        local_gemm.nvfp4_localcta_gemm(
            A_q.fp4,
            A_q.sc,
            A_q.sg,
            B_q.fp4,
            B_q.sc,
            B_q.sg,
            out,
        )
        return out

    tk = _get_tk_nvfp4()
    config_id = _select_nvfp4_gemm_config(M, N, K)
    if config_id is not None:
        tk.nvfp4_gemm_config(A_q.fp4, A_q.sc, A_q.sg, B_q.fp4, B_q.sc, B_q.sg, out, int(config_id))
    else:
        tk.nvfp4_gemm(A_q.fp4, A_q.sc, A_q.sg, B_q.fp4, B_q.sc, B_q.sg, out)
    return out


def _nvfp4_chunked_logits_chunk() -> int:
    return max(int(os.environ.get("FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_CHUNK", "8192")), 256)


def _use_nvfp4_chunked_logits_g_cache(M: int, V: int) -> bool:
    raw = os.environ.get("FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE", "auto").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw != "auto":
        return False
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        device = torch.cuda.current_device()
        reserved_bytes = torch.cuda.memory_reserved(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
    except Exception:
        return False
    reclaimable_cache_bytes = max(int(reserved_bytes) - int(allocated_bytes), 0)
    available_bytes = int(free_bytes) + reclaimable_cache_bytes
    logits_bytes = int(M) * int(V) * 2
    # PyTorch can normally satisfy the direct logits/G path from cached blocks
    # retained after the first fast step. Raw cudaMemGetInfo() ignores that
    # cache and incorrectly forces the much slower chunked path.
    return logits_bytes > int(0.75 * available_bytes)


def _nvfp4_vocab_parallel_chunked_logits_chunk() -> int:
    raw = os.environ.get(
        "FP4_CCE_V4_NVFP4_VOCAB_PARALLEL_CHUNKED_LOGITS_G_CACHE_CHUNK",
        os.environ.get(
            "FP4_CCE_V4_NVFP4_VOCAB_PARALLEL_CHUNKED_LOGITS_CHUNK",
            os.environ.get("FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_CHUNK", "4096"),
        ),
    )
    return max(int(raw), 256)


def _use_nvfp4_vocab_parallel_chunked_logits_g_cache(M: int, V: int) -> bool:
    raw = os.environ.get(
        "FP4_CCE_V4_NVFP4_VOCAB_PARALLEL_CHUNKED_LOGITS_G_CACHE",
        "0",
    ).strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw != "auto":
        return False
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        device = torch.cuda.current_device()
        reserved_bytes = torch.cuda.memory_reserved(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
    except Exception:
        return False
    reclaimable_cache_bytes = max(int(reserved_bytes) - int(allocated_bytes), 0)
    available_bytes = int(free_bytes) + reclaimable_cache_bytes
    logits_bytes = int(M) * int(V) * 2
    return logits_bytes > int(0.75 * available_bytes)


def _use_nvfp4_vocab_parallel_chunked_recompute() -> bool:
    return os.environ.get(
        "FP4_CCE_V4_NVFP4_VOCAB_PARALLEL_CHUNKED_RECOMPUTE",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}


def _nvfp4_vocab_parallel_chunked_logits_g_cache(
    q_x: NVFP4Quantized,
    q_w: NVFP4Quantized,
    targets: torch.Tensor,
    ignore_index: int,
    global_vocab_size: int,
    vocab_start: int,
    tp_group,
    encode_centric: bool,
):
    if not encode_centric or not _use_nvfp4_g_constant_scale():
        raise RuntimeError("chunked vocab-parallel NVFP4 G-cache requires encode-centric constant-scale G")

    M = int(q_x.fp4.shape[0])
    V = int(q_w.fp4.shape[0])
    device = q_x.fp4.device
    chunk = _nvfp4_vocab_parallel_chunked_logits_chunk()
    chunk = ((chunk + 255) // 256) * 256
    chunk = max(256, min(chunk, V))

    valid = targets.ne(int(ignore_index))
    local_targets = targets.to(torch.long) - int(vocab_start)
    local_valid_cols = max(min(int(global_vocab_size) - int(vocab_start), V), 0)
    use_dist = (
        tp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=tp_group) > 1
    )

    local_max = torch.full((M,), -float("inf"), dtype=torch.float32, device=device)
    target_logits = torch.full((M,), -float("inf"), dtype=torch.float32, device=device)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        local_max = torch.maximum(local_max, logits_f.max(dim=1).values)
        in_range = (
            valid
            & (local_targets >= start)
            & (local_targets < min(end, local_valid_cols))
            & (targets < int(global_vocab_size))
        )
        if bool(in_range.any()):
            rows = torch.where(in_range)[0]
            cols = local_targets[rows] - start
            target_logits[rows] = logits_f[rows, cols]
        del logits, logits_f

    global_max = local_max.clone()
    if use_dist:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
        dist.all_reduce(target_logits, op=dist.ReduceOp.MAX, group=tp_group)

    local_den = torch.zeros((M,), dtype=torch.float32, device=device)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        local_den.add_(torch.exp(logits_f - global_max[:, None]).sum(dim=1))
        del logits, logits_f

    global_den = local_den.clone()
    if use_dist:
        dist.all_reduce(global_den, op=dist.ReduceOp.SUM, group=tp_group)
    global_lse = global_max + torch.log(global_den)
    denom = valid.sum().clamp(min=1)
    row_loss = torch.where(valid, global_lse - target_logits, torch.zeros_like(global_lse))
    loss = row_loss.sum() / denom

    row_fp4 = torch.empty((M, V // 2), dtype=torch.float4_e2m1fn_x2, device=device)
    row_sc = torch.empty((M // 128, V // 64, 512), dtype=torch.float8_e4m3fn, device=device)
    col_fp4 = torch.empty((V, M // 2), dtype=torch.float4_e2m1fn_x2, device=device)
    col_sc = torch.empty((V // 128, M // 64, 512), dtype=torch.float8_e4m3fn, device=device)
    row_fp4_u8 = row_fp4.view(torch.uint8)
    col_fp4_u8 = col_fp4.view(torch.uint8)
    sg = None
    quantize_g = quantize_nvfp4_row_and_col_tk_constant_scale

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        grad = torch.exp(logits_f - global_lse[:, None])
        in_range = (
            valid
            & (local_targets >= start)
            & (local_targets < min(end, local_valid_cols))
            & (targets < int(global_vocab_size))
        )
        if bool(in_range.any()):
            rows = torch.where(in_range)[0]
            cols = local_targets[rows] - start
            grad[rows, cols] -= 1.0
        grad.masked_fill_(~valid[:, None], 0.0)
        g_row_q, g_col_q = quantize_g(grad.to(torch.bfloat16), encode_centric=encode_centric)
        row_fp4_u8[:, start // 2 : end // 2].copy_(g_row_q.fp4.view(torch.uint8))
        row_sc[:, start // 64 : end // 64].copy_(g_row_q.sc)
        col_fp4_u8[start:end].copy_(g_col_q.fp4.view(torch.uint8))
        col_sc[start // 128 : end // 128].copy_(g_col_q.sc)
        sg = g_row_q.sg
        del logits, logits_f, grad, g_row_q, g_col_q

    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def _nvfp4_vocab_parallel_chunked_loss_lse(
    q_x: NVFP4Quantized,
    q_w: NVFP4Quantized,
    targets: torch.Tensor,
    ignore_index: int,
    global_vocab_size: int,
    vocab_start: int,
    tp_group,
):
    M = int(q_x.fp4.shape[0])
    V = int(q_w.fp4.shape[0])
    device = q_x.fp4.device
    chunk = _nvfp4_vocab_parallel_chunked_logits_chunk()
    chunk = ((chunk + 255) // 256) * 256
    chunk = max(256, min(chunk, V))

    valid = targets.ne(int(ignore_index))
    local_targets = targets.to(torch.long) - int(vocab_start)
    local_valid_cols = max(min(int(global_vocab_size) - int(vocab_start), V), 0)
    use_dist = (
        tp_group is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size(group=tp_group) > 1
    )

    local_max = torch.full((M,), -float("inf"), dtype=torch.float32, device=device)
    target_logits = torch.full((M,), -float("inf"), dtype=torch.float32, device=device)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        del logits
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        local_max = torch.maximum(local_max, logits_f.max(dim=1).values)
        in_range = (
            valid
            & (local_targets >= start)
            & (local_targets < min(end, local_valid_cols))
            & (targets < int(global_vocab_size))
        )
        if bool(in_range.any()):
            rows = torch.where(in_range)[0]
            cols = local_targets[rows] - start
            target_logits[rows] = logits_f[rows, cols]
        del logits_f

    global_max = local_max.clone()
    if use_dist:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
        dist.all_reduce(target_logits, op=dist.ReduceOp.MAX, group=tp_group)

    local_den = torch.zeros((M,), dtype=torch.float32, device=device)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        del logits
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        logits_f.sub_(global_max[:, None])
        logits_f.exp_()
        local_den.add_(logits_f.sum(dim=1))
        del logits_f

    global_den = local_den.clone()
    if use_dist:
        dist.all_reduce(global_den, op=dist.ReduceOp.SUM, group=tp_group)
    global_lse = global_max + torch.log(global_den)
    denom = valid.sum().clamp(min=1)
    row_loss = torch.where(valid, global_lse - target_logits, torch.zeros_like(global_lse))
    loss = row_loss.sum() / denom
    return loss, global_lse


def _nvfp4_vocab_parallel_chunked_recompute_backward(ctx, grad_output):
    (
        x_row_fp4,
        x_row_sc,
        x_row_sg,
        x_col_fp4,
        x_col_sc,
        x_col_sg,
        w_row_fp4,
        w_row_sc,
        w_row_sg,
        w_col_fp4,
        w_col_sc,
        w_col_sg,
        targets,
        global_lse,
    ) = ctx.saved_tensors

    n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
    scale = (grad_output.float() / n_valid).reshape(())
    q_x = NVFP4Quantized(x_row_fp4, x_row_sc, x_row_sg)
    x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
    q_w_sg = w_row_sg
    w_col_sg = w_col_sg

    M = int(x_row_fp4.shape[0])
    V = int(w_row_fp4.shape[0])
    K = int(x_col_fp4.shape[0])
    device = x_row_fp4.device
    chunk = _nvfp4_vocab_parallel_chunked_logits_chunk()
    chunk = ((chunk + 255) // 256) * 256
    chunk = max(256, min(chunk, V))
    local_targets = targets.to(torch.long) - int(ctx.vocab_start)
    valid = targets.ne(ctx.ignore_index)
    local_valid_cols = int(ctx.local_valid_cols)

    quantize_g = quantize_nvfp4_row_and_col_tk_constant_scale
    dE = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
    dC = torch.empty((V, K), dtype=torch.bfloat16, device=device)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            w_row_fp4[start:end],
            w_row_sc[start // 128 : end // 128],
            q_w_sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        del logits
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        logits_f.sub_(global_lse[:, None])
        logits_f.exp_()
        in_range = (
            valid
            & (local_targets >= start)
            & (local_targets < min(end, local_valid_cols))
            & (targets < int(ctx.global_vocab_size))
        )
        if bool(in_range.any()):
            rows = torch.where(in_range)[0]
            cols = local_targets[rows] - start
            logits_f[rows, cols] -= 1.0
        logits_f.masked_fill_(~valid[:, None], 0.0)
        grad_bf16 = logits_f.to(torch.bfloat16)
        del logits_f
        g_row_q, g_col_q = quantize_g(
            grad_bf16,
            encode_centric=bool(ctx.encode_centric),
        )
        del grad_bf16

        g_row = NVFP4Quantized(g_row_q.fp4, g_row_q.sc, g_row_q.sg * scale)
        w_col_fp4_chunk = (
            w_col_fp4.view(torch.uint8)[:, start // 2 : end // 2]
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
        )
        w_col_chunk = NVFP4Quantized(
            w_col_fp4_chunk,
            w_col_sc[:, start // 64 : end // 64].contiguous(),
            w_col_sg,
        )
        dE.add_(tk_nvfp4_gemm(g_row, w_col_chunk))

        g_col = NVFP4Quantized(g_col_q.fp4, g_col_q.sc, g_col_q.sg * scale)
        dC[start:end].copy_(tk_nvfp4_gemm(g_col, x_col_q))
        del g_row_q, g_col_q, g_row, g_col, w_col_chunk, q_w_chunk

    if ctx.reduce_dE:
        torch.distributed.all_reduce(dE, group=ctx.tp_group)
    return dE, dC, None, None, None, None, None, None, None


def _nvfp4_chunked_logits_g_cache(
    q_x: NVFP4Quantized,
    q_w: NVFP4Quantized,
    targets: torch.Tensor,
    ignore_index: int,
    vocab_size: int,
    assume_all_valid: bool,
    encode_centric: bool,
):
    if not encode_centric or not _use_nvfp4_g_constant_scale():
        raise RuntimeError("chunked NVFP4 logits G-cache requires encode-centric constant-scale G")

    M = int(q_x.fp4.shape[0])
    V = int(q_w.fp4.shape[0])
    device = q_x.fp4.device
    chunk = _nvfp4_chunked_logits_chunk()
    chunk = ((chunk + 255) // 256) * 256
    chunk = max(256, min(chunk, V))

    valid = torch.ones_like(targets, dtype=torch.bool, device=device) if assume_all_valid else targets.ne(ignore_index)
    row_max = torch.full((M,), -float("inf"), dtype=torch.float32, device=device)
    target_logits = torch.zeros((M,), dtype=torch.float32, device=device)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if vocab_size < end:
            logits_f[:, max(0, vocab_size - start) :] = -float("inf")
        row_max = torch.maximum(row_max, logits_f.max(dim=1).values)
        in_range = valid & (targets >= start) & (targets < end)
        if bool(in_range.any()):
            rows = torch.where(in_range)[0]
            cols = targets[rows] - start
            target_logits[rows] = logits_f[rows, cols]
        del logits, logits_f

    exp_sum = torch.zeros((M,), dtype=torch.float32, device=device)
    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if vocab_size < end:
            logits_f[:, max(0, vocab_size - start) :] = -float("inf")
        exp_sum.add_(torch.exp(logits_f - row_max[:, None]).sum(dim=1))
        del logits, logits_f

    lse = row_max + torch.log(exp_sum)
    n_valid = targets.numel() if assume_all_valid else valid.sum().clamp(min=1)
    loss = ((lse - target_logits) * valid.float()).sum() / n_valid.float()

    row_fp4 = torch.empty((M, V // 2), dtype=torch.float4_e2m1fn_x2, device=device)
    row_sc = torch.empty((M // 128, V // 64, 512), dtype=torch.float8_e4m3fn, device=device)
    col_fp4 = torch.empty((V, M // 2), dtype=torch.float4_e2m1fn_x2, device=device)
    col_sc = torch.empty((V // 128, M // 64, 512), dtype=torch.float8_e4m3fn, device=device)
    row_fp4_u8 = row_fp4.view(torch.uint8)
    col_fp4_u8 = col_fp4.view(torch.uint8)
    sg = None
    quantize_g = quantize_nvfp4_row_and_col_tk_constant_scale

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = NVFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
            q_w.sg,
        )
        logits = tk_nvfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if vocab_size < end:
            logits_f[:, max(0, vocab_size - start) :] = -float("inf")
        grad = torch.exp(logits_f - lse[:, None])
        in_range = valid & (targets >= start) & (targets < end)
        if bool(in_range.any()):
            rows = torch.where(in_range)[0]
            cols = targets[rows] - start
            grad[rows, cols] -= 1.0
        grad.masked_fill_(~valid[:, None], 0.0)
        g_row_q, g_col_q = quantize_g(grad.to(torch.bfloat16), encode_centric=encode_centric)
        row_fp4_u8[:, start // 2 : end // 2].copy_(g_row_q.fp4.view(torch.uint8))
        row_sc[:, start // 64 : end // 64].copy_(g_row_q.sc)
        col_fp4_u8[start:end].copy_(g_col_q.fp4.view(torch.uint8))
        col_sc[start // 128 : end // 128].copy_(g_col_q.sc)
        sg = g_row_q.sg
        del logits, logits_f, grad, g_row_q, g_col_q

    return loss, row_fp4, row_sc, sg, col_fp4, col_sc, sg


def _nvfp4_p_cache_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
    vocab_size: int,
    encode_centric: bool = True,
):
    encode_centric = bool(encode_centric)
    p_data_sr = _use_nvfp4_p_data_sr()
    p_target_split = _use_nvfp4_p_target_split()
    p_topk_split = _nvfp4_p_topk_split()
    if p_topk_split and not p_target_split:
        raise RuntimeError("NVFP4 P top-k split also requires target split")
    if not encode_centric:
        if p_target_split or p_topk_split:
            raise RuntimeError(
                "NVFP4 P target/top-k split requires encode-centric quantization"
            )
        loss, probs = loss_and_probs(logits, targets, valid, int(ignore_index))
        quantize_p = _select_nvfp4_p_quantizer()
        p_row_q, p_col_q = quantize_p(probs, encode_centric=False)
        return (
            loss,
            p_row_q.fp4,
            p_row_q.sc,
            p_row_q.sg,
            p_col_q.fp4,
            p_col_q.sc,
            p_col_q.sg,
            None,
            None,
            None,
        )

    if (p_data_sr or p_target_split or p_topk_split) and (
        use_tma_nvfp4_p_cache() or use_tiled_nvfp4_p_cache()
    ):
        raise RuntimeError(
            "NVFP4 P data SR and target split require the staged CUDA softmax "
            "plus TK quantizer route; disable the TMA/tiled P producer"
        )
    if use_tma_nvfp4_p_cache():
        result = nvfp4_tma_p_cache(
            logits,
            targets,
            valid,
            int(vocab_size),
        )
        return (*result, None, None, None)
    if use_tiled_nvfp4_p_cache():
        result = nvfp4_tiled_p_cache(
            logits,
            targets,
            valid,
            int(vocab_size),
            _use_nvfp4_p_constant_scale(),
        )
        return (*result, None, None, None)
    if use_staged_nvfp4_p_cache():
        if (
            _use_nvfp4_p_constant_scale()
            and _use_nvfp4_fused_staged_p_cache()
            and not p_data_sr
            and not p_target_split
        ):
            result = nvfp4_staged_p_cache(
                logits,
                targets,
                valid,
                int(vocab_size),
            )
            return (*result, None, None, None)
        if p_topk_split == 4:
            (
                loss,
                probs,
                target_probs,
                topk_probs,
                topk_indices,
            ) = direct_loss_and_probs_target_top4_split(
                logits, targets, valid, int(vocab_size)
            )
        elif p_topk_split == 1:
            (
                loss,
                probs,
                target_probs,
                topk_probs,
                topk_indices,
            ) = direct_loss_and_probs_target_top1_split(
                logits, targets, valid, int(vocab_size)
            )
        elif p_target_split:
            loss, probs, target_probs = direct_loss_and_probs_target_split(
                logits, targets, valid, int(vocab_size)
            )
            topk_probs = None
            topk_indices = None
        else:
            loss, probs = direct_loss_and_probs(
                logits, targets, valid, int(vocab_size)
            )
            target_probs = None
            topk_probs = None
            topk_indices = None
        p_row_q, p_col_q = _select_nvfp4_p_quantizer()(probs, encode_centric=True)
        return (
            loss,
            p_row_q.fp4,
            p_row_q.sc,
            p_row_q.sg,
            p_col_q.fp4,
            p_col_q.sc,
            p_col_q.sg,
            target_probs,
            topk_indices,
            topk_probs,
        )

    if p_target_split:
        raise RuntimeError("NVFP4 P target split requires the staged CUDA softmax")
    loss, probs = loss_and_probs(logits, targets, valid, int(ignore_index))
    p_row_q, p_col_q = _select_nvfp4_p_quantizer()(probs, encode_centric=True)
    return (
        loss,
        p_row_q.fp4,
        p_row_q.sc,
        p_row_q.sg,
        p_col_q.fp4,
        p_col_q.sc,
        p_col_q.sg,
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
class NVFP4CCE_TK_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e_q_fp4, e_q_sc, e_q_sg,
                c_q_fp4, c_q_sc, c_q_sg,
                c_bf16, e_bf16,
                targets, ignore_index):
        tk = _get_tk_nvfp4()

        M = e_q_fp4.shape[0]
        N = c_q_fp4.shape[0]

        # TK NVFP4 GEMM: logits(M, N) = E @ C^T
        logits = torch.zeros(M, N, dtype=torch.bfloat16, device=e_q_fp4.device)
        tk.nvfp4_gemm(e_q_fp4, e_q_sc, e_q_sg, c_q_fp4, c_q_sc, c_q_sg, logits)

        # Compute LSE for chunked backward (avoid storing full logits)
        lse = logits.float().logsumexp(dim=-1)  # (M,)

        loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)

        ctx.save_for_backward(c_bf16, e_bf16, targets, lse)
        ctx.ignore_index = ignore_index

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        c_bf16, e_bf16, targets, lse = ctx.saved_tensors

        M, K = e_bf16.shape
        V = c_bf16.shape[0]
        device = e_bf16.device

        # Precompute masking
        valid = targets.ne(ctx.ignore_index)
        n_valid = valid.sum().clamp(min=1).float()
        grad_scale = (grad_output / n_valid).item()

        # Quantize E and C for the FP4 backward kernel
        e_q = quantize_nvfp4_tk(e_bf16, keep_bf16=False)
        c_q = quantize_nvfp4_tk(c_bf16, keep_bf16=False)

        # Fused backward kernel: FP4 GEMM recomputes logits, consumer
        # computes softmax gradient = exp(logits*scale - lse) - 1[target]
        # Output: BF16 grad_logits (M, V)
        bwd = _get_nvfp4_cce_backward()
        grad_logits = torch.zeros(M, V, dtype=torch.bfloat16, device=device)
        bwd.backward_L4_SG8(
            e_q.fp4, e_q.sc, e_q.sg,
            c_q.fp4, c_q.sc, c_q.sg,
            grad_logits, lse, targets,
            grad_scale, M, V
        )

        # BF16 matmul for dE and dC
        dE = (grad_logits.float() @ c_bf16.float()).to(torch.bfloat16)
        dC = (grad_logits.float().T @ e_bf16.float()).to(torch.bfloat16)

        return None, None, None, None, None, None, dC, dE, None, None


class NVFP4CCE_PCache_Function(torch.autograd.Function):
    """v4 CCE prototype: materialize logits once, then cache quantized P.

    Backward computes:
      dE = scale * P @ C - scale * C[target]
      dC = scale * P.T @ E - scale * scatter_add(E, target)
    where P is stored in row and transpose NVFP4 form from the forward pass.
    """

    @staticmethod
    def forward(ctx, x, weight, targets, ignore_index, vocab_size, encode_centric):
        encode_centric = bool(encode_centric)
        if _use_nvfp4_group_input_quant() and encode_centric:
            q_x, q_x_col, q_w, q_w_col = quantize_both_nvfp4_row_and_col_tk(
                x, weight, encode_centric=True
            )
        else:
            quantize_x = _select_nvfp4_xw_quantizer("X")
            quantize_w = _select_nvfp4_xw_quantizer("W")
            q_x, q_x_col = quantize_x(x, encode_centric=encode_centric)
            q_w, q_w_col = quantize_w(weight, encode_centric=encode_centric)
        logits = (
            bf16_logits_cuda(x, weight)
            if _use_nvfp4_bf16_logits()
            else tk_nvfp4_gemm(q_x, q_w)
        )
        if vocab_size < logits.shape[1]:
            logits = logits.clone()
            logits[:, vocab_size:] = -float("inf")
        if _use_nvfp4_exact_target_logit():
            replace_target_logits_bf16(
                logits,
                x,
                weight,
                targets,
                int(ignore_index),
                int(vocab_size),
            )
        assume_all_valid = assume_all_valid_full_vocab(logits, int(vocab_size))
        valid = None

        def get_valid():
            nonlocal valid
            if valid is None:
                valid = targets.ne(ignore_index)
            return valid

        use_g_cache = _use_nvfp4_g_cache()
        g_target_split = use_g_cache and _use_nvfp4_g_target_split()
        g_topk_split = _nvfp4_g_topk_split() if use_g_cache else 0
        if _use_nvfp4_exact_selected_logits() and not g_target_split:
            raise RuntimeError(
                "exact target/top-k repair requires NVFP4 G target split"
            )
        if g_topk_split and not g_target_split:
            raise RuntimeError("NVFP4 G top-k split also requires target split")
        g_top1_split = False
        g_top2_split = False
        g_top4_split = False
        g_top6_split = False
        g_wide_topk_split = False
        g_row_normalization = torch.empty(
            0, dtype=torch.float32, device=x.device
        )
        use_bf16_p_dweight = _use_nvfp4_bf16_p_dweight()
        p_target_split = False
        p_top1_split = False
        p_top4_split = False
        if use_g_cache:
            if g_target_split:
                g_cache_outputs = _nvfp4_target_split_g_cache(
                    logits,
                    x,
                    weight,
                    targets,
                    get_valid(),
                    int(vocab_size),
                    encode_centric=encode_centric,
                )
                (
                    loss,
                    g_row_fp4,
                    g_row_sc,
                    g_row_sg,
                    g_col_fp4,
                    g_col_sc,
                    g_col_sg,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    *delayed_row_normalization,
                ) = g_cache_outputs
                if delayed_row_normalization:
                    g_row_normalization = delayed_row_normalization[0]
                g_top1_split = (
                    topk_indices is not None and topk_indices.dim() == 1
                )
                g_top2_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] == 2
                )
                g_top4_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] == 4
                )
                g_top6_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] == 6
                )
                g_wide_topk_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] in (8, 12, 16)
                )
            elif _use_nvfp4_fused_g_cache() and encode_centric:
                fused_impl = _nvfp4_fused_g_cache_impl()
                if fused_impl == "direct":
                    loss, grad_probs = direct_loss_and_grad_probs(
                        logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                    )
                    g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                        grad_probs, encode_centric=encode_centric
                    )
                    g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                    g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
                elif _use_nvfp4_staged_g_cache() or fused_impl == "staged":
                    loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = nvfp4_staged_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
                elif fused_impl == "tiled":
                    loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = nvfp4_tiled_g_cache(
                        logits,
                        targets,
                        get_valid(),
                        int(vocab_size),
                        global_scale_max=_nvfp4_g_scale_max(),
                        block_scale=_use_nvfp4_g_chunk_scale(),
                    )
                else:
                    loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = nvfp4_tma_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
            elif _use_nvfp4_direct_g_producer() or not encode_centric:
                loss, grad_probs = direct_loss_and_grad_probs(
                    logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                )
                g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                    grad_probs, encode_centric=encode_centric
                )
                g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
            else:
                loss, grad_probs = direct_loss_and_probs(logits, targets, get_valid(), int(vocab_size))
                grad_probs = grad_probs.clone()
                valid_now = get_valid()
                if bool(valid_now.all()):
                    rows = torch.arange(targets.numel(), device=targets.device)
                    grad_probs[rows, targets] -= 1
                else:
                    rows = torch.where(valid_now)[0]
                    grad_probs[rows, targets[rows]] -= 1
                g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                    grad_probs, encode_centric=encode_centric
                )
                g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
            if g_target_split:
                saved = [
                    g_row_fp4, g_row_sc, g_row_sg,
                    g_col_fp4, g_col_sc, g_col_sg,
                    q_x_col.fp4, q_x_col.sc, q_x_col.sg,
                    q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                    x, weight, targets, target_probs, g_row_normalization,
                ]
                if (
                    g_top1_split
                    or g_top2_split
                    or g_top4_split
                    or g_top6_split
                    or g_wide_topk_split
                ):
                    saved.extend((topk_indices, topk_probs))
                ctx.save_for_backward(*saved)
            else:
                ctx.save_for_backward(
                    g_row_fp4, g_row_sc, g_row_sg,
                    g_col_fp4, g_col_sc, g_col_sg,
                    q_x_col.fp4, q_x_col.sc, q_x_col.sg,
                    q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                    targets,
                )
        elif use_bf16_p_dweight:
            loss, probs = direct_loss_and_probs(logits, targets, get_valid(), int(vocab_size))
            p_row_q = quantize_nvfp4_tk(
                probs, keep_bf16=False, encode_centric=encode_centric
            )
            ctx.save_for_backward(
                p_row_q.fp4, p_row_q.sc, p_row_q.sg,
                q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                x, weight, targets, probs,
            )
        else:
            (
                loss,
                p_row_fp4,
                p_row_sc,
                p_row_sg,
                p_col_fp4,
                p_col_sc,
                p_col_sg,
                target_probs,
                topk_indices,
                topk_probs,
            ) = _nvfp4_p_cache_from_logits(
                logits,
                targets,
                get_valid(),
                int(ignore_index),
                int(vocab_size),
                encode_centric=encode_centric,
            )
            p_target_split = target_probs is not None
            p_top1_split = topk_indices is not None and topk_indices.dim() == 1
            p_top4_split = topk_indices is not None and topk_indices.dim() == 2
            saved = [
                p_row_fp4, p_row_sc, p_row_sg,
                p_col_fp4, p_col_sc, p_col_sg,
                q_x_col.fp4, q_x_col.sc, q_x_col.sg,
                q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                x, weight, targets,
            ]
            if p_target_split:
                saved.append(target_probs)
            if p_top1_split or p_top4_split:
                saved.extend((topk_indices, topk_probs))
            ctx.save_for_backward(*saved)
        ctx.ignore_index = int(ignore_index)
        ctx.use_g_cache = bool(use_g_cache)
        ctx.g_target_split = bool(g_target_split)
        ctx.g_top1_split = bool(g_top1_split)
        ctx.g_top2_split = bool(g_top2_split)
        ctx.g_top4_split = bool(g_top4_split)
        ctx.g_top6_split = bool(g_top6_split)
        ctx.g_wide_topk_split = bool(g_wide_topk_split)
        ctx.g_delayed_row_normalization = bool(g_row_normalization.numel())
        ctx.use_bf16_p_dweight = bool(use_bf16_p_dweight)
        ctx.p_target_split = bool(p_target_split)
        ctx.p_top1_split = bool(p_top1_split)
        ctx.p_top4_split = bool(p_top4_split)
        ctx.assume_all_valid = bool(assume_all_valid)
        ctx.backward_v5_requant = bool(_use_nvfp4_backward_v5_requant())
        ctx.g_localcta_constant_layout = bool(
            ctx.use_g_cache
            and _use_nvfp4_localcta_v4_quant()
            and not ctx.backward_v5_requant
        )
        ctx.encode_centric = bool(encode_centric)
        if ctx.backward_v5_requant and not (
            (ctx.use_g_cache and ctx.g_target_split)
            or (not ctx.use_g_cache and ctx.p_target_split)
        ):
            raise RuntimeError(
                "FP4_CCE_V4_NVFP4_BACKWARD_V5_REQUANT requires P or G target split"
            )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_g_cache:
            if ctx.g_target_split:
                if (
                    ctx.g_top1_split
                    or ctx.g_top2_split
                    or ctx.g_top4_split
                    or ctx.g_top6_split
                    or ctx.g_wide_topk_split
                ):
                    (
                        g_row_fp4, g_row_sc, g_row_sg,
                        g_col_fp4, g_col_sc, g_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs, g_row_normalization,
                        topk_indices, topk_probs,
                    ) = ctx.saved_tensors
                else:
                    (
                        g_row_fp4, g_row_sc, g_row_sg,
                        g_col_fp4, g_col_sc, g_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs, g_row_normalization,
                    ) = ctx.saved_tensors
            else:
                (
                    g_row_fp4, g_row_sc, g_row_sg,
                    g_col_fp4, g_col_sc, g_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    w_col_fp4, w_col_sc, w_col_sg,
                    targets,
                ) = ctx.saved_tensors
        elif ctx.use_bf16_p_dweight:
            (
                p_row_fp4, p_row_sc, p_row_sg,
                w_col_fp4, w_col_sc, w_col_sg,
                x, weight, targets, probs,
            ) = ctx.saved_tensors
        else:
            if ctx.p_target_split:
                if ctx.p_top1_split or ctx.p_top4_split:
                    (
                        p_row_fp4, p_row_sc, p_row_sg,
                        p_col_fp4, p_col_sc, p_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs,
                        topk_indices, topk_probs,
                    ) = ctx.saved_tensors
                else:
                    (
                        p_row_fp4, p_row_sc, p_row_sg,
                        p_col_fp4, p_col_sc, p_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs,
                    ) = ctx.saved_tensors
            else:
                (
                    p_row_fp4, p_row_sc, p_row_sg,
                    p_col_fp4, p_col_sc, p_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    w_col_fp4, w_col_sc, w_col_sg,
                    x, weight, targets,
                ) = ctx.saved_tensors

        if getattr(ctx, "assume_all_valid", False):
            scale = (grad_output.float() / float(targets.numel())).reshape(())
        else:
            n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
            scale = (grad_output.float() / n_valid).reshape(())

        if getattr(ctx, "backward_v5_requant", False):
            x_col_q, w_col_q = _nvfp4_backward_v5_cols(
                x,
                weight,
                encode_centric=ctx.encode_centric,
            )
            x_col_fp4, x_col_sc, x_col_sg = (
                x_col_q.fp4,
                x_col_q.sc,
                x_col_q.sg,
            )
            w_col_fp4, w_col_sc, w_col_sg = (
                w_col_q.fp4,
                w_col_q.sc,
                w_col_q.sg,
            )

        if ctx.use_g_cache:
            g_row = _scaled_g_cache_operand(
                g_row_fp4,
                g_row_sc,
                g_row_sg,
                scale,
                localcta_constant_layout=bool(
                    getattr(ctx, "g_localcta_constant_layout", False)
                ),
            )
            g_col = _scaled_g_cache_operand(
                g_col_fp4,
                g_col_sc,
                g_col_sg,
                scale,
                localcta_constant_layout=bool(
                    getattr(ctx, "g_localcta_constant_layout", False)
                ),
            )
            x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
            w_col_q = NVFP4Quantized(w_col_fp4, w_col_sc, w_col_sg)
            dE = tk_nvfp4_gemm(g_row, w_col_q)
            if getattr(ctx, "g_delayed_row_normalization", False):
                dE.mul_(g_row_normalization[:, None])
            repair_stream = None
            if (
                ctx.g_top1_split
                or ctx.g_top2_split
                or ctx.g_top4_split
                or ctx.g_top6_split
                or ctx.g_wide_topk_split
            ):
                repair_stream = _launch_overlapped_topk_dE_repair(
                    dE,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            dC = tk_nvfp4_gemm(g_col, x_col_q)
            if repair_stream is not None:
                _finish_overlapped_topk_repair(
                    repair_stream,
                    dC,
                    x,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_wide_topk_split:
                sparse_correct_target_topk_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top6_split:
                sparse_correct_target_top6_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top4_split:
                sparse_correct_target_top4_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top2_split:
                sparse_correct_target_top2_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top1_split:
                sparse_correct_target_top1_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_target_split:
                sparse_correct_target_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    scale,
                    ctx.ignore_index,
                )
            return dE, dC, None, None, None, None

        p_row_sg_scaled = p_row_sg * scale
        p_row = NVFP4Quantized(p_row_fp4, p_row_sc, p_row_sg_scaled)
        w_col_q = NVFP4Quantized(w_col_fp4, w_col_sc, w_col_sg)

        dE = tk_nvfp4_gemm(p_row, w_col_q)
        if ctx.use_bf16_p_dweight:
            V = weight.shape[0]
            K = x.shape[1]
            dC = torch.empty(V, K, dtype=torch.bfloat16, device=x.device)
            chunk = _nvfp4_bf16_p_dweight_chunk()
            for start in range(0, V, chunk):
                end = min(start + chunk, V)
                dC[start:end] = probs[:, start:end].transpose(0, 1) @ x
            dC.mul_(scale)
        else:
            p_col_sg_scaled = p_col_sg * scale
            p_col = NVFP4Quantized(p_col_fp4, p_col_sc, p_col_sg_scaled)
            x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
            dC = tk_nvfp4_gemm(p_col, x_col_q)

        if ctx.p_top4_split:
            sparse_correct_target_top4_split(
                dE,
                dC,
                x,
                weight,
                targets,
                target_probs,
                topk_indices,
                topk_probs,
                scale,
                ctx.ignore_index,
            )
        elif ctx.p_top1_split:
            sparse_correct_target_top1_split(
                dE,
                dC,
                x,
                weight,
                targets,
                target_probs,
                topk_indices,
                topk_probs,
                scale,
                ctx.ignore_index,
            )
        elif ctx.p_target_split:
            sparse_correct_target_split(
                dE,
                dC,
                x,
                weight,
                targets,
                target_probs,
                scale,
                ctx.ignore_index,
            )
        else:
            sparse_correct(dE, dC, x, weight, targets, scale, ctx.ignore_index)

        return dE, dC, None, None, None, None


class NVFP4CCE_PCache_PrequantX_Function(torch.autograd.Function):
    """v4 CCE with x/x.T quantization supplied by an upstream producer."""

    @staticmethod
    def forward(
        ctx,
        x,
        x_fp4,
        x_sc,
        x_sg,
        x_col_fp4,
        x_col_sc,
        x_col_sg,
        weight,
        targets,
        ignore_index,
        vocab_size,
        encode_centric,
        forward_format,
        prequantized_weight,
    ):
        encode_centric = bool(encode_centric)
        forward_format = int(forward_format)
        if forward_format not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                "forward format must be 0=NVFP4, 1=MXFP8, 2=MXFP4, "
                "3=direct FP8, 4=MXFP6 E2M3, or 5=MXFP6 E3M2"
            )
        use_mxfp8_forward = forward_format == 1
        use_mxfp4_forward = forward_format == 2
        use_direct_fp8_forward = forward_format == 3
        use_mxfp6_forward = forward_format in (4, 5)
        mxfp6_format = "e2m3" if forward_format == 4 else "e3m2"
        use_mxfp8_g_cache = _use_mxfp8_g_cache()
        use_mxfp4_g_cache = _use_mxfp4_g_cache()
        use_mixed_g_cache = _use_mxfp8_row_nvfp4_col_g_cache()
        use_mixed_dw_mxfp8_cols = _use_mixed_dw_mxfp8_cols()
        use_lowp_logits_bf16_dhidden = _use_lowp_logits_bf16_dhidden()
        use_lowp_logits_bf16_dhidden_inplace_g = (
            _use_lowp_logits_bf16_dhidden_inplace_g()
        )
        use_lowp_logits_bf16_dweight = _use_lowp_logits_bf16_dweight()
        use_lowp_logits_bf16_cache_elision = bool(
            use_lowp_logits_bf16_dhidden
            and use_lowp_logits_bf16_dhidden_inplace_g
            and use_lowp_logits_bf16_dweight
        )
        if (
            use_lowp_logits_bf16_dhidden_inplace_g
            and not use_lowp_logits_bf16_dhidden
        ):
            raise RuntimeError(
                "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN_INPLACE_G=1 requires "
                "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1"
            )
        if use_lowp_logits_bf16_dweight and not use_lowp_logits_bf16_dhidden:
            raise RuntimeError(
                "FP4_CCE_V4_LOWP_LOGITS_BF16_DWEIGHT=1 requires "
                "FP4_CCE_V4_LOWP_LOGITS_BF16_DHIDDEN=1"
            )
        if sum((use_mxfp8_g_cache, use_mxfp4_g_cache, use_mixed_g_cache)) > 1:
            raise RuntimeError(
                "MXFP8, MXFP4, and mixed G-cache modes are mutually exclusive"
            )
        if use_mxfp8_g_cache and not use_mxfp8_forward:
            raise RuntimeError("MXFP8 G-cache requires MXFP8 forward operands")
        if use_mixed_g_cache and not (
            use_mxfp4_forward or use_mxfp8_forward
        ):
            raise RuntimeError(
                "mixed MXFP8-row/NVFP4-column G-cache requires MXFP4 or "
                "MXFP8 forward operands"
            )
        if use_mixed_dw_mxfp8_cols and not (
            use_mixed_g_cache and use_mxfp8_forward
        ):
            raise RuntimeError(
                "FP4_CCE_V4_MIXED_DW_MXFP8_COLS requires the mixed G-cache "
                "and MXFP8 forward operands"
            )
        pure_nvfp4_tiled_g = bool(
            not use_mxfp8_g_cache
            and not use_mxfp4_g_cache
            and not use_mixed_g_cache
            and _use_nvfp4_g_cache()
            and _use_nvfp4_g_target_split()
            and _use_nvfp4_fused_g_cache()
            and _nvfp4_fused_g_cache_impl() == "tiled"
        )
        if (
            _mxfp8_logit_temperature() != 1.0
            and not use_mxfp4_g_cache
            and not use_mixed_g_cache
            and not pure_nvfp4_tiled_g
        ):
            raise RuntimeError(
                "logit temperature requires MXFP4 G-cache or the pure-NVFP4 "
                "tiled target-split G-cache"
            )
        if use_mxfp8_g_cache or use_mixed_dw_mxfp8_cols:
            q_x_col = MXFP8Quantized(x_col_fp4, x_col_sc)
        elif use_mxfp4_g_cache:
            q_x_col = MXFP4Quantized(x_col_fp4, x_col_sc)
        else:
            q_x_col = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)

        q_w = None
        q_w_col = None
        if prequantized_weight is not None:
            if not isinstance(prequantized_weight, tuple) or len(prequantized_weight) != 2:
                raise TypeError("prequantized_weight must be a (row, column) tuple")
            q_w, q_w_col = prequantized_weight

        if use_direct_fp8_forward:
            if not use_mxfp4_g_cache:
                raise RuntimeError(
                    "direct FP8 forward currently requires MXFP4 G-cache"
            )
            q_x = DirectFP8Quantized(x_fp4, x_sc)
            if q_w is None:
                q_w, q_w_col = quantize_direct_fp8_row_mxfp4_col(
                    weight, role="W"
                )
            elif not isinstance(q_w, DirectFP8Quantized) or not isinstance(
                q_w_col, MXFP4Quantized
            ):
                raise TypeError(
                    "direct FP8 forward requires cached DirectFP8 row and "
                    "MXFP4 column weight operands"
                )
        elif use_mxfp8_forward:
            if (
                _use_nvfp4_bf16_logits()
                and not _use_analysis_bf16_logits_with_mxfp8_backward()
            ):
                raise RuntimeError(
                    "MXFP8 forward and BF16 logits are mutually exclusive; "
                    "the analysis-only backward-format gate is required"
                )
            if (
                os.environ.get("FP4_CCE_V4_NVFP4_W_DATA_SR", "0") != "0"
                or os.environ.get("FP4_CCE_V4_NVFP4_W_SCALE_SR", "0") != "0"
            ):
                raise RuntimeError(
                    "the fused MXFP8/NVFP4 weight producer does not support "
                    "NVFP4 weight stochastic rounding"
                )
            q_x = MXFP8Quantized(x_fp4, x_sc)
            if q_w is not None:
                expected_col_type = (
                    MXFP8Quantized
                    if use_mxfp8_g_cache or use_mixed_g_cache
                    else MXFP4Quantized
                    if use_mxfp4_g_cache
                    else NVFP4Quantized
                )
                if not isinstance(q_w, MXFP8Quantized) or not isinstance(
                    q_w_col, expected_col_type
                ):
                    raise TypeError(
                        "MXFP8 forward received incompatible cached weight operands"
                    )
            elif use_mxfp4_g_cache:
                q_w, q_w_col = quantize_mxfp8_row_mxfp4_col(weight)
            elif use_mixed_g_cache:
                q_w, q_w_col = quantize_mxfp8_row_and_col_fused(weight)
            else:
                q_w, q_w_col = quantize_mxfp8_row_nvfp4_col_localcta_v4(
                    weight,
                    encode_centric=encode_centric,
                    four_over_six_mae=_use_nvfp4_w_four_over_six_mae(),
                )
        elif use_mxfp6_forward:
            if not use_mxfp4_g_cache:
                raise RuntimeError("MXFP6 forward currently requires MXFP4 G-cache")
            if _use_nvfp4_bf16_logits():
                raise RuntimeError(
                    "MXFP6 forward and BF16 logits are mutually exclusive"
                )
            q_x = MXFP6Quantized(x_fp4, x_sc, format=mxfp6_format)
            if q_w is not None:
                if not isinstance(q_w, MXFP6Quantized) or not isinstance(
                    q_w_col, MXFP4Quantized
                ):
                    raise TypeError(
                        "MXFP6 forward requires cached MXFP6 row and MXFP4 "
                        "column weight operands"
                    )
            else:
                q_w, q_w_col = quantize_mxfp6_row_mxfp4_col(
                    weight, format=mxfp6_format
                )
        elif use_mxfp4_forward:
            if _use_nvfp4_bf16_logits():
                raise RuntimeError(
                    "MXFP4 forward and BF16 logits are mutually exclusive"
                )
            q_x = MXFP4Quantized(x_fp4, x_sc)
            if q_w is not None:
                expected_col_type = (
                    MXFP8Quantized
                    if use_mixed_g_cache
                    else MXFP4Quantized
                    if use_mxfp4_g_cache
                    else NVFP4Quantized
                )
                if not isinstance(q_w, MXFP4Quantized) or not isinstance(
                    q_w_col, expected_col_type
                ):
                    raise TypeError(
                        "MXFP4 forward received incompatible cached weight operands"
                    )
            elif use_mixed_g_cache:
                q_w, q_w_col = quantize_mxfp4_row_mxfp8_col(weight)
            elif use_mxfp4_g_cache:
                q_w, q_w_col = quantize_mxfp4_row_and_col_tk(
                    weight, mode=1, role="W"
                )
            else:
                q_w, q_w_col = quantize_mxfp4_row_nvfp4_col_v5(
                    weight,
                    encode_centric=encode_centric,
                    role="W",
                )
        else:
            q_x = NVFP4Quantized(x_fp4, x_sc, x_sg)
            if q_w is not None:
                expected_col_type = (
                    MXFP4Quantized if use_mxfp4_g_cache else NVFP4Quantized
                )
                if not isinstance(q_w, NVFP4Quantized) or not isinstance(
                    q_w_col, expected_col_type
                ):
                    raise TypeError(
                        "NVFP4 forward received incompatible cached weight operands"
                    )
            elif use_mxfp4_g_cache and _use_nvfp4_localcta_v4_quant():
                q_w = quantize_nvfp4_row_localcta_v4(
                    weight,
                    encode_centric=encode_centric,
                    four_over_six_mae=_use_nvfp4_w_four_over_six_mae(),
                )
                q_w_col = quantize_mxfp4_col_tk(weight, mode=1)
            elif use_mxfp4_g_cache:
                q_w = quantize_nvfp4_tk(
                    weight,
                    keep_bf16=False,
                    encode_centric=encode_centric,
                )
                q_w_col = quantize_mxfp4_col_tk(weight, mode=1)
            else:
                quantize_weight = _select_nvfp4_xw_quantizer("W")
                q_w, q_w_col = quantize_weight(
                    weight, encode_centric=encode_centric
                )

        w_col_data = (
            q_w_col.fp8 if isinstance(q_w_col, MXFP8Quantized) else q_w_col.fp4
        )
        w_col_sc = q_w_col.sc
        w_col_sg = (
            torch.empty(0, dtype=torch.float32, device=weight.device)
            if isinstance(q_w_col, (MXFP8Quantized, MXFP4Quantized))
            else q_w_col.sg
        )

        use_g_cache = _use_nvfp4_g_cache()
        g_target_split = use_g_cache and _use_nvfp4_g_target_split()
        g_topk_split = _nvfp4_g_topk_split() if use_g_cache else 0
        if use_mxfp8_g_cache and not (use_g_cache and g_target_split):
            raise RuntimeError("MXFP8 G-cache requires G-cache target splitting")
        if use_mxfp4_g_cache and not use_g_cache:
            raise RuntimeError("MXFP4 G-cache requires G-cache mode")
        if use_mixed_g_cache and not (use_g_cache and g_target_split):
            raise RuntimeError(
                "mixed MXFP8/NVFP4 G-cache requires G-cache target splitting"
            )
        if _use_nvfp4_exact_selected_logits() and not g_target_split:
            raise RuntimeError(
                "exact target/top-k repair requires NVFP4 G target split"
            )
        if g_topk_split and not g_target_split:
            raise RuntimeError("NVFP4 G top-k split also requires target split")
        g_top1_split = False
        g_top2_split = False
        g_top4_split = False
        g_top6_split = False
        g_wide_topk_split = False
        g_prepared_dC = None
        g_preparation_stream = None
        g_valid_count = None
        if use_lowp_logits_bf16_dhidden:
            _validate_lowp_logits_bf16_dhidden(
                use_mxfp8_forward=use_mxfp8_forward,
                use_mixed_g_cache=use_mixed_g_cache,
                use_mixed_dw_mxfp8_cols=use_mixed_dw_mxfp8_cols,
                g_target_split=g_target_split,
                g_topk_split=g_topk_split,
            )
        if use_lowp_logits_bf16_cache_elision:
            if not isinstance(q_x_col, MXFP8Quantized) or not isinstance(
                q_w_col, MXFP8Quantized
            ):
                raise RuntimeError(
                    "BF16-both cache elision requires typed MXFP8 column sentinels"
                )
            if any(
                tensor.numel()
                for tensor in (
                    q_x_col.fp8,
                    q_x_col.sc,
                    q_w_col.fp8,
                    q_w_col.sc,
                )
            ):
                raise RuntimeError(
                    "BF16-both cache elision requires omitted X/W column operands"
                )
        g_row_normalization = torch.empty(
            0, dtype=torch.float32, device=x.device
        )
        bf16_dhidden_from_lowp_g = torch.empty(
            0, dtype=torch.bfloat16, device=x.device
        )
        bf16_dweight_from_lowp_g = torch.empty(
            0, dtype=torch.bfloat16, device=x.device
        )
        use_bf16_p_dweight = _use_nvfp4_bf16_p_dweight()
        p_target_split = False
        p_top1_split = False
        p_top4_split = False
        assume_all_valid = (
            os.environ.get("FP4_CCE_V4_SOFTMAX_ASSUME_ALL_VALID_FULL_VOCAB", "0") != "0"
            and int(vocab_size) == int(weight.shape[0])
        )
        use_chunked_g_cache = (
            use_g_cache
            and not use_mxfp8_forward
            and not use_mxfp4_forward
            and not use_direct_fp8_forward
            and not use_mxfp6_forward
            and not use_mxfp8_g_cache
            and not use_mxfp4_g_cache
            and not use_mixed_g_cache
            and _use_nvfp4_chunked_logits_g_cache(int(q_x.fp4.shape[0]), int(q_w.fp4.shape[0]))
            and encode_centric
            and _use_nvfp4_g_constant_scale()
        )
        if _use_nvfp4_bf16_logits() and use_chunked_g_cache:
            raise RuntimeError(
                "NVFP4 BF16 logits require the direct full-logits path; disable "
                "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE"
            )
        if g_target_split and use_chunked_g_cache:
            raise RuntimeError(
                "NVFP4 G target split requires the direct full-logits producer; "
                "disable FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE"
            )
        if use_chunked_g_cache:
            loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = (
                _nvfp4_chunked_logits_g_cache(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(vocab_size),
                    bool(assume_all_valid),
                    encode_centric=encode_centric,
                )
            )
            ctx.save_for_backward(
                g_row_fp4, g_row_sc, g_row_sg,
                g_col_fp4, g_col_sc, g_col_sg,
                x_col_fp4, x_col_sc, x_col_sg,
                q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                targets,
            )
            ctx.ignore_index = int(ignore_index)
            ctx.use_g_cache = True
            ctx.g_mxfp8 = False
            ctx.g_mxfp4 = False
            ctx.g_target_split = False
            ctx.g_top1_split = False
            ctx.g_top2_split = False
            ctx.g_top4_split = False
            ctx.g_top6_split = False
            ctx.g_wide_topk_split = False
            ctx.use_bf16_p_dweight = False
            ctx.p_target_split = False
            ctx.p_top1_split = False
            ctx.p_top4_split = False
            ctx.assume_all_valid = bool(assume_all_valid)
            ctx.g_localcta_constant_layout = bool(
                _use_nvfp4_localcta_v4_quant()
                and _use_nvfp4_g_constant_scale()
            )
            return loss

        logit_centers = None
        use_centered_mxfp8_logits = (
            use_mxfp8_forward
            and _env_flag("FP4_CCE_V4_MXFP8_CENTERED_FP8_LOGITS", False)
        )
        if use_centered_mxfp8_logits:
            if _env_flag("FP4_CCE_V4_MXFP8_FP8_LOGITS", False):
                raise RuntimeError(
                    "plain and centered MXFP8 E4M3 logit modes are mutually "
                    "exclusive"
                )
            if not use_mxfp4_g_cache:
                raise RuntimeError(
                    "centered MXFP8 logits currently require MXFP4 G-cache"
                )
            if not _use_nvfp4_g_target_split():
                raise RuntimeError(
                    "centered MXFP8 logits require target/top-k splitting"
                )
            if _use_nvfp4_exact_target_logit():
                raise RuntimeError(
                    "centered MXFP8 logits require exact-selected repair instead "
                    "of in-place exact-target replacement"
                )

        if use_direct_fp8_forward:
            logits = direct_fp8_gemm(q_x, q_w)
        elif use_mxfp8_forward:
            if _use_analysis_bf16_logits_with_mxfp8_backward():
                if not _use_nvfp4_bf16_logits():
                    raise RuntimeError(
                        "the analysis-only MXFP8 backward gate requires "
                        "FP4_CCE_V4_NVFP4_BF16_LOGITS=1"
                    )
                logits = bf16_logits_cuda(x, weight)
            elif use_centered_mxfp8_logits:
                logits, logit_centers = tk_mxfp8_gemm_centered(q_x, q_w)
            else:
                logits = tk_mxfp8_gemm(
                    q_x,
                    q_w,
                    fp8_output=_env_flag(
                        "FP4_CCE_V4_MXFP8_FP8_LOGITS", False
                    ),
                )
        elif use_mxfp6_forward:
            logits = tk_mxfp6_gemm(q_x, q_w)
        elif use_mxfp4_forward:
            logits = tk_mxfp4_gemm(q_x, q_w)
        elif _use_nvfp4_bf16_logits():
            logits = bf16_logits_cuda(x, weight)
        else:
            logits = tk_nvfp4_gemm(q_x, q_w)
        if vocab_size < logits.shape[1]:
            logits = logits.clone()
            logits[:, vocab_size:] = -float("inf")
        if _use_nvfp4_exact_target_logit():
            replace_target_logits_bf16(
                logits,
                x,
                weight,
                targets,
                int(ignore_index),
                int(vocab_size),
            )
        assume_all_valid = assume_all_valid_full_vocab(logits, int(vocab_size))
        valid = None

        def get_valid():
            nonlocal valid, g_valid_count
            if valid is None:
                if use_mxfp4_g_cache or use_mixed_g_cache:
                    valid, g_valid_count = valid_mask_count_cuda(
                        targets, int(ignore_index)
                    )
                else:
                    valid = targets.ne(ignore_index)
            return valid

        if use_g_cache:
            if g_target_split:
                if use_mixed_g_cache:
                    mixed_cache = _mxfp8_row_nvfp4_col_target_split_g_cache(
                        logits,
                        x,
                        weight,
                        targets,
                        get_valid(),
                        int(vocab_size),
                        return_lse=use_lowp_logits_bf16_dhidden,
                    )
                    if use_lowp_logits_bf16_dhidden:
                        (
                            loss,
                            g_row_fp4,
                            g_row_sc,
                            g_col_fp4,
                            g_col_sc,
                            g_col_sg,
                            target_probs,
                            topk_indices,
                            topk_probs,
                            g_preparation_stream,
                            g_row_normalization,
                            corrected_lse,
                        ) = mixed_cache
                        # The fused producer has already replaced target/top-k
                        # logits with -inf and repaired LSE using their exact
                        # BF16 dot products.  Scale the remaining approximate
                        # logits by the production temperature, materialize
                        # the BF16 tail gradient, restore exact selected
                        # coefficients in that same matrix, and immediately
                        # reduce it against the original BF16 classifier.  It
                        # is important that selected and tail terms share one
                        # FP32-accumulating GEMM: adding selected dE afterward
                        # measurably amplifies final-norm gamma cancellation.
                        temperature = _mxfp8_logit_temperature()
                        repaired_g_producer = (
                            softmax_repaired_grad_probs_from_lse_inplace
                            if use_lowp_logits_bf16_dhidden_inplace_g
                            else softmax_repaired_grad_probs_from_lse
                        )
                        bf16_tail_g = repaired_g_producer(
                            logits,
                            corrected_lse,
                            targets,
                            get_valid(),
                            target_probs,
                            topk_indices,
                            topk_probs,
                            int(vocab_size),
                            temperature,
                        )
                        bf16_dhidden_from_lowp_g = bf16_tail_g @ weight
                        if use_lowp_logits_bf16_dweight:
                            bf16_dweight_from_lowp_g = (
                                bf16_tail_g.transpose(0, 1) @ x
                            )
                        del bf16_tail_g
                    else:
                        (
                            loss,
                            g_row_fp4,
                            g_row_sc,
                            g_col_fp4,
                            g_col_sc,
                            g_col_sg,
                            target_probs,
                            topk_indices,
                            topk_probs,
                            g_preparation_stream,
                            *delayed_row_normalization,
                        ) = mixed_cache
                        if delayed_row_normalization:
                            g_row_normalization = delayed_row_normalization[0]
                    del mixed_cache
                    g_row_sg = torch.empty(
                        0, dtype=torch.float32, device=logits.device
                    )
                elif use_mxfp8_g_cache:
                    (
                        loss,
                        g_row_fp4,
                        g_row_sc,
                        g_col_fp4,
                        g_col_sc,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        g_preparation_stream,
                    ) = _mxfp8_target_split_g_cache(
                        logits,
                        x,
                        weight,
                        targets,
                        get_valid(),
                        int(vocab_size),
                    )
                    g_row_sg = torch.empty(
                        0, dtype=torch.float32, device=logits.device
                    )
                    g_col_sg = g_row_sg
                elif use_mxfp4_g_cache:
                    (
                        loss,
                        g_row_fp4,
                        g_row_sc,
                        g_col_fp4,
                        g_col_sc,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        g_prepared_dC,
                        g_preparation_stream,
                        *delayed_row_normalization,
                    ) = _mxfp4_target_split_g_cache(
                        logits,
                        x,
                        weight,
                        targets,
                        get_valid(),
                        int(ignore_index),
                        int(vocab_size),
                        logit_centers=logit_centers,
                    )
                    if delayed_row_normalization:
                        g_row_normalization = delayed_row_normalization[0]
                    g_row_sg = torch.empty(
                        0, dtype=torch.float32, device=logits.device
                    )
                    g_col_sg = g_row_sg
                else:
                    g_cache_outputs = _nvfp4_target_split_g_cache(
                        logits,
                        x,
                        weight,
                        targets,
                        get_valid(),
                        int(vocab_size),
                        encode_centric=encode_centric,
                    )
                    (
                        loss,
                        g_row_fp4,
                        g_row_sc,
                        g_row_sg,
                        g_col_fp4,
                        g_col_sc,
                        g_col_sg,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        *delayed_row_normalization,
                    ) = g_cache_outputs
                    if delayed_row_normalization:
                        g_row_normalization = delayed_row_normalization[0]
                    if (
                        g_topk_split
                        and _use_sparse_repair_overlap()
                        and _use_mx_compact_dw_repair()
                    ):
                        g_preparation_stream = _sparse_dC_preparation_stream(
                            logits.device
                        )
                        if g_preparation_stream is not None:
                            g_preparation_stream.wait_stream(
                                torch.cuda.current_stream(logits.device)
                            )
                g_top1_split = (
                    topk_indices is not None and topk_indices.dim() == 1
                )
                g_top2_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] == 2
                )
                g_top4_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] == 4
                )
                g_top6_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] == 6
                )
                g_wide_topk_split = (
                    topk_indices is not None
                    and topk_indices.dim() == 2
                    and topk_indices.shape[1] in (8, 12, 16)
                )
            elif use_mxfp4_g_cache:
                (
                    loss,
                    g_row_fp4,
                    g_row_sc,
                    g_col_fp4,
                    g_col_sc,
                ) = mxfp4_tiled_g_cache(
                    logits,
                    targets,
                    get_valid(),
                    int(vocab_size),
                )
                g_row_sg = torch.empty(
                    0, dtype=torch.float32, device=logits.device
                )
                g_col_sg = g_row_sg
            elif _use_nvfp4_fused_g_cache() and encode_centric:
                fused_impl = _nvfp4_fused_g_cache_impl()
                if fused_impl == "direct":
                    loss, grad_probs = direct_loss_and_grad_probs(
                        logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                    )
                    g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                        grad_probs, encode_centric=encode_centric
                    )
                    g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                    g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
                elif _use_nvfp4_staged_g_cache() or fused_impl == "staged":
                    loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = nvfp4_staged_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
                elif fused_impl == "tiled":
                    loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = nvfp4_tiled_g_cache(
                        logits,
                        targets,
                        get_valid(),
                        int(vocab_size),
                        global_scale_max=_nvfp4_g_scale_max(),
                        block_scale=_use_nvfp4_g_chunk_scale(),
                    )
                else:
                    loss, g_row_fp4, g_row_sc, g_row_sg, g_col_fp4, g_col_sc, g_col_sg = nvfp4_tma_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
            elif _use_nvfp4_direct_g_producer() or not encode_centric:
                loss, grad_probs = direct_loss_and_grad_probs(
                    logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                )
                g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                    grad_probs, encode_centric=encode_centric
                )
                g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
            else:
                loss, grad_probs = direct_loss_and_probs(logits, targets, get_valid(), int(vocab_size))
                grad_probs = grad_probs.clone()
                valid_now = get_valid()
                if bool(valid_now.all()):
                    rows = torch.arange(targets.numel(), device=targets.device)
                    grad_probs[rows, targets] -= 1
                else:
                    rows = torch.where(valid_now)[0]
                    grad_probs[rows, targets[rows]] -= 1
                g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                    grad_probs, encode_centric=encode_centric
                )
                g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
            if use_lowp_logits_bf16_cache_elision:
                # The fused G-row producer remains mandatory: it owns the
                # production loss/top-16 repair and the checkpointed SR
                # advancement.  Once the one in-place BF16 G has fed both
                # exact-gradient GEMMs, however, none of the four lowp
                # backward operands may survive into autograd.
                (
                    g_row_fp4,
                    g_row_sc,
                    g_row_sg,
                    g_col_fp4,
                    g_col_sc,
                    g_col_sg,
                    x_col_fp4,
                    x_col_sc,
                    x_col_sg,
                    w_col_data,
                    w_col_sc,
                    w_col_sg,
                ) = _empty_mxfp8_backward_cache_tensors(x.device)
                g_row_normalization = torch.empty(
                    0, dtype=torch.float32, device=x.device
                )
            if g_target_split:
                saved = [
                    g_row_fp4, g_row_sc, g_row_sg,
                    g_col_fp4, g_col_sc, g_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    w_col_data, w_col_sc, w_col_sg,
                    x, weight, targets, target_probs, g_row_normalization,
                    bf16_dhidden_from_lowp_g,
                    bf16_dweight_from_lowp_g,
                ]
                if (
                    g_top1_split
                    or g_top2_split
                    or g_top4_split
                    or g_top6_split
                    or g_wide_topk_split
                ):
                    saved.extend((topk_indices, topk_probs))
                    if g_prepared_dC is not None:
                        saved.extend(g_prepared_dC)
                ctx.save_for_backward(*saved)
            else:
                ctx.save_for_backward(
                    g_row_fp4, g_row_sc, g_row_sg,
                    g_col_fp4, g_col_sc, g_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    w_col_data, w_col_sc, w_col_sg,
                    targets,
                )
        elif use_bf16_p_dweight:
            loss, probs = direct_loss_and_probs(logits, targets, get_valid(), int(vocab_size))
            p_row_q = quantize_nvfp4_tk(
                probs, keep_bf16=False, encode_centric=encode_centric
            )
            ctx.save_for_backward(
                p_row_q.fp4, p_row_q.sc, p_row_q.sg,
                q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                x, weight, targets, probs,
            )
        else:
            (
                loss,
                p_row_fp4,
                p_row_sc,
                p_row_sg,
                p_col_fp4,
                p_col_sc,
                p_col_sg,
                target_probs,
                topk_indices,
                topk_probs,
            ) = _nvfp4_p_cache_from_logits(
                logits,
                targets,
                get_valid(),
                int(ignore_index),
                int(vocab_size),
                encode_centric=encode_centric,
            )
            p_target_split = target_probs is not None
            p_top1_split = topk_indices is not None and topk_indices.dim() == 1
            p_top4_split = topk_indices is not None and topk_indices.dim() == 2
            saved = [
                p_row_fp4, p_row_sc, p_row_sg,
                p_col_fp4, p_col_sc, p_col_sg,
                x_col_fp4, x_col_sc, x_col_sg,
                q_w_col.fp4, q_w_col.sc, q_w_col.sg,
                x, weight, targets,
            ]
            if p_target_split:
                saved.append(target_probs)
            if p_top1_split or p_top4_split:
                saved.extend((topk_indices, topk_probs))
            ctx.save_for_backward(*saved)
        ctx.ignore_index = int(ignore_index)
        ctx.use_g_cache = bool(use_g_cache)
        ctx.g_mxfp8 = bool(use_mxfp8_g_cache)
        ctx.g_mxfp4 = bool(use_mxfp4_g_cache)
        ctx.g_mxfp4_atbt_dw = bool(
            use_mxfp4_g_cache and _use_mxfp4_g_atbt_dw()
        )
        ctx.g_mixed_mxfp8_row_nvfp4_col = bool(use_mixed_g_cache)
        ctx.g_mixed_dw_mxfp8_cols = bool(use_mixed_dw_mxfp8_cols)
        ctx.g_target_split = bool(g_target_split)
        ctx.g_top1_split = bool(g_top1_split)
        ctx.g_top2_split = bool(g_top2_split)
        ctx.g_top4_split = bool(g_top4_split)
        ctx.g_top6_split = bool(g_top6_split)
        ctx.g_wide_topk_split = bool(g_wide_topk_split)
        ctx.g_prepared_dC = g_prepared_dC is not None
        ctx.g_preparation_stream = g_preparation_stream
        ctx.g_valid_count = g_valid_count
        ctx.g_delayed_row_normalization = bool(g_row_normalization.numel())
        ctx.lowp_logits_bf16_dhidden = bool(
            use_lowp_logits_bf16_dhidden
        )
        ctx.lowp_logits_bf16_dweight = bool(
            use_lowp_logits_bf16_dweight
        )
        ctx.lowp_logits_bf16_cache_elision = bool(
            use_lowp_logits_bf16_cache_elision
        )
        ctx.use_bf16_p_dweight = bool(use_bf16_p_dweight)
        ctx.p_target_split = bool(p_target_split)
        ctx.p_top1_split = bool(p_top1_split)
        ctx.p_top4_split = bool(p_top4_split)
        ctx.assume_all_valid = bool(assume_all_valid)
        ctx.backward_v5_requant = bool(_use_nvfp4_backward_v5_requant())
        ctx.g_localcta_constant_layout = bool(
            ctx.use_g_cache
            and not ctx.g_mxfp8
            and not ctx.g_mxfp4
            and _use_nvfp4_localcta_v4_quant()
            and _use_nvfp4_g_constant_scale()
            and not ctx.backward_v5_requant
        )
        ctx.encode_centric = bool(encode_centric)
        if ctx.g_mxfp8 and ctx.backward_v5_requant:
            raise RuntimeError("MXFP8 G-cache does not use NVFP4 backward requantization")
        if ctx.g_mxfp4 and ctx.backward_v5_requant:
            raise RuntimeError("MXFP4 G-cache does not use NVFP4 backward requantization")
        if ctx.g_mixed_mxfp8_row_nvfp4_col and ctx.backward_v5_requant:
            raise RuntimeError(
                "mixed MXFP8/NVFP4 G-cache does not use backward requantization"
            )
        if ctx.backward_v5_requant and not (
            (ctx.use_g_cache and ctx.g_target_split)
            or (not ctx.use_g_cache and ctx.p_target_split)
        ):
            raise RuntimeError(
                "FP4_CCE_V4_NVFP4_BACKWARD_V5_REQUANT requires P or G target split"
            )
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        need_dE = bool(ctx.needs_input_grad[0])
        need_dC = bool(ctx.needs_input_grad[7])
        if ctx.use_g_cache:
            if ctx.g_target_split:
                if (
                    ctx.g_top1_split
                    or ctx.g_top2_split
                    or ctx.g_top4_split
                    or ctx.g_top6_split
                    or ctx.g_wide_topk_split
                ):
                    if ctx.g_prepared_dC:
                        (
                            g_row_fp4, g_row_sc, g_row_sg,
                            g_col_fp4, g_col_sc, g_col_sg,
                            x_col_fp4, x_col_sc, x_col_sg,
                            w_col_fp4, w_col_sc, w_col_sg,
                            x, weight, targets, target_probs,
                            g_row_normalization,
                            bf16_dhidden_from_lowp_g,
                            bf16_dweight_from_lowp_g,
                            topk_indices, topk_probs,
                            prepared_vocab_rows,
                            prepared_x_rows,
                            prepared_coefficients,
                        ) = ctx.saved_tensors
                    else:
                        (
                            g_row_fp4, g_row_sc, g_row_sg,
                            g_col_fp4, g_col_sc, g_col_sg,
                            x_col_fp4, x_col_sc, x_col_sg,
                            w_col_fp4, w_col_sc, w_col_sg,
                            x, weight, targets, target_probs,
                            g_row_normalization,
                            bf16_dhidden_from_lowp_g,
                            bf16_dweight_from_lowp_g,
                            topk_indices, topk_probs,
                        ) = ctx.saved_tensors
                else:
                    (
                        g_row_fp4, g_row_sc, g_row_sg,
                        g_col_fp4, g_col_sc, g_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs,
                        g_row_normalization,
                        bf16_dhidden_from_lowp_g,
                        bf16_dweight_from_lowp_g,
                    ) = ctx.saved_tensors
            else:
                (
                    g_row_fp4, g_row_sc, g_row_sg,
                    g_col_fp4, g_col_sc, g_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    w_col_fp4, w_col_sc, w_col_sg,
                    targets,
                ) = ctx.saved_tensors
        elif ctx.use_bf16_p_dweight:
            (
                p_row_fp4, p_row_sc, p_row_sg,
                w_col_fp4, w_col_sc, w_col_sg,
                x, weight, targets, probs,
            ) = ctx.saved_tensors
        else:
            if ctx.p_target_split:
                if ctx.p_top1_split or ctx.p_top4_split:
                    (
                        p_row_fp4, p_row_sc, p_row_sg,
                        p_col_fp4, p_col_sc, p_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs,
                        topk_indices, topk_probs,
                    ) = ctx.saved_tensors
                else:
                    (
                        p_row_fp4, p_row_sc, p_row_sg,
                        p_col_fp4, p_col_sc, p_col_sg,
                        x_col_fp4, x_col_sc, x_col_sg,
                        w_col_fp4, w_col_sc, w_col_sg,
                        x, weight, targets, target_probs,
                    ) = ctx.saved_tensors
            else:
                (
                    p_row_fp4, p_row_sc, p_row_sg,
                    p_col_fp4, p_col_sc, p_col_sg,
                    x_col_fp4, x_col_sc, x_col_sg,
                    w_col_fp4, w_col_sc, w_col_sg,
                    x, weight, targets,
                ) = ctx.saved_tensors

        if getattr(ctx, "g_valid_count", None) is not None:
            scale = backward_scale_cuda(
                grad_output.float().reshape(1), ctx.g_valid_count
            ).reshape(())
        elif getattr(ctx, "assume_all_valid", False):
            scale = (grad_output.float() / float(targets.numel())).reshape(())
        else:
            n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
            scale = (grad_output.float() / n_valid).reshape(())

        if getattr(ctx, "backward_v5_requant", False):
            x_col_q, w_col_q = _nvfp4_backward_v5_cols(
                x,
                weight,
                encode_centric=ctx.encode_centric,
            )
            x_col_fp4, x_col_sc, x_col_sg = (
                x_col_q.fp4,
                x_col_q.sc,
                x_col_q.sg,
            )
            w_col_fp4, w_col_sc, w_col_sg = (
                w_col_q.fp4,
                w_col_q.sc,
                w_col_q.sg,
            )

        compact_dw = bool(
            ctx.use_g_cache
            and not getattr(ctx, "lowp_logits_bf16_dweight", False)
            and _use_sparse_repair_overlap()
            and _use_mx_compact_dw_repair()
            and (
                ctx.g_top1_split
                or ctx.g_top2_split
                or ctx.g_top4_split
                or ctx.g_top6_split
                or ctx.g_wide_topk_split
            )
        )
        compact_repair = None
        mxfp4_dE_stream = None

        if ctx.use_g_cache:
            if getattr(ctx, "g_mixed_mxfp8_row_nvfp4_col", False):
                if getattr(ctx, "lowp_logits_bf16_cache_elision", False):
                    if any(
                        tensor.numel()
                        for tensor in (
                            g_row_fp4,
                            g_row_sc,
                            g_row_sg,
                            g_col_fp4,
                            g_col_sc,
                            g_col_sg,
                            x_col_fp4,
                            x_col_sc,
                            x_col_sg,
                            w_col_fp4,
                            w_col_sc,
                            w_col_sg,
                        )
                    ):
                        raise RuntimeError(
                            "BF16-both cache elision restored a dead lowp cache"
                        )
                g_row = MXFP8Quantized(g_row_fp4, g_row_sc)
                if getattr(ctx, "g_mixed_dw_mxfp8_cols", False):
                    g_col = MXFP8Quantized(g_col_fp4, g_col_sc)
                    x_col_q = MXFP8Quantized(x_col_fp4, x_col_sc)
                else:
                    g_col = _scaled_g_cache_operand(
                        g_col_fp4,
                        g_col_sc,
                        g_col_sg,
                        scale,
                        localcta_constant_layout=bool(
                            getattr(ctx, "g_localcta_constant_layout", False)
                        ),
                    )
                    x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
                w_col_q = MXFP8Quantized(w_col_fp4, w_col_sc)
                if (
                    need_dE
                    and not getattr(ctx, "lowp_logits_bf16_dhidden", False)
                    and _use_mx_backward_gemm_overlap()
                ):
                    mxfp4_dE_stream = _mx_backward_gemm_stream(x.device)
                if not need_dE:
                    dE = None
                elif getattr(ctx, "lowp_logits_bf16_dhidden", False):
                    if bf16_dhidden_from_lowp_g.numel() != x.numel():
                        raise RuntimeError(
                            "missing saved BF16 dHidden for the lowp-logit path"
                        )
                    dE = bf16_dhidden_from_lowp_g.mul(scale)
                elif mxfp4_dE_stream is None:
                    dE = tk_mxfp8_gemm(g_row, w_col_q)
                    dE.mul_(scale)
                    if getattr(ctx, "g_delayed_row_normalization", False):
                        dE.mul_(g_row_normalization[:, None])
                else:
                    current_stream = torch.cuda.current_stream(x.device)
                    mxfp4_dE_stream.wait_stream(current_stream)
                    with torch.cuda.stream(mxfp4_dE_stream):
                        dE = tk_mxfp8_gemm(g_row, w_col_q)
                        dE.mul_(scale)
                        if getattr(ctx, "g_delayed_row_normalization", False):
                            dE.mul_(g_row_normalization[:, None])
                    for tensor in (
                        g_row.fp8,
                        g_row.sc,
                        w_col_q.fp8,
                        w_col_q.sc,
                        scale,
                    ):
                        tensor.record_stream(mxfp4_dE_stream)
                    if getattr(ctx, "g_delayed_row_normalization", False):
                        g_row_normalization.record_stream(mxfp4_dE_stream)
                    dE.record_stream(mxfp4_dE_stream)
            elif ctx.g_mxfp8:
                g_row = MXFP8Quantized(g_row_fp4, g_row_sc)
                g_col = MXFP8Quantized(g_col_fp4, g_col_sc)
                x_col_q = MXFP8Quantized(x_col_fp4, x_col_sc)
                w_col_q = MXFP8Quantized(w_col_fp4, w_col_sc)
                dE = tk_mxfp8_gemm(g_row, w_col_q)
                dE.mul_(scale)
            elif ctx.g_mxfp4:
                g_row = MXFP4Quantized(g_row_fp4, g_row_sc)
                g_col = (
                    None
                    if ctx.g_mxfp4_atbt_dw
                    else MXFP4Quantized(g_col_fp4, g_col_sc)
                )
                x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
                w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)
                if _use_mx_backward_gemm_overlap():
                    mxfp4_dE_stream = _mx_backward_gemm_stream(x.device)
                if mxfp4_dE_stream is None:
                    dE = tk_mxfp4_gemm(g_row, w_col_q, output_scale=scale)
                    if getattr(ctx, "g_delayed_row_normalization", False):
                        dE.mul_(g_row_normalization[:, None])
                else:
                    current_stream = torch.cuda.current_stream(x.device)
                    mxfp4_dE_stream.wait_stream(current_stream)
                    with torch.cuda.stream(mxfp4_dE_stream):
                        dE = tk_mxfp4_gemm(
                            g_row, w_col_q, output_scale=scale
                        )
                        if getattr(ctx, "g_delayed_row_normalization", False):
                            dE.mul_(g_row_normalization[:, None])
                    for tensor in (
                        g_row.fp4,
                        g_row.sc,
                        w_col_q.fp4,
                        w_col_q.sc,
                        scale,
                    ):
                        tensor.record_stream(mxfp4_dE_stream)
                    if getattr(ctx, "g_delayed_row_normalization", False):
                        g_row_normalization.record_stream(mxfp4_dE_stream)
                    dE.record_stream(mxfp4_dE_stream)
            else:
                g_row = _scaled_g_cache_operand(
                    g_row_fp4,
                    g_row_sc,
                    g_row_sg,
                    scale,
                    localcta_constant_layout=bool(
                        getattr(ctx, "g_localcta_constant_layout", False)
                    ),
                )
                g_col = _scaled_g_cache_operand(
                    g_col_fp4,
                    g_col_sc,
                    g_col_sg,
                    scale,
                    localcta_constant_layout=bool(
                        getattr(ctx, "g_localcta_constant_layout", False)
                    ),
                )
                x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
                w_col_q = NVFP4Quantized(w_col_fp4, w_col_sc, w_col_sg)
                dE = tk_nvfp4_gemm(g_row, w_col_q)
                if getattr(ctx, "g_delayed_row_normalization", False):
                    dE.mul_(g_row_normalization[:, None])
            if compact_dw:
                compact_repair = _launch_compact_topk_dC_repair(
                    ctx.g_preparation_stream,
                    x,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    int(weight.shape[0]),
                    ctx.ignore_index,
                )
            has_topk_repair = bool(
                ctx.g_top1_split
                or ctx.g_top2_split
                or ctx.g_top4_split
                or ctx.g_top6_split
                or ctx.g_wide_topk_split
            )
            if not need_dE and not (
                getattr(ctx, "g_mixed_mxfp8_row_nvfp4_col", False)
                and getattr(ctx, "g_mixed_dw_mxfp8_cols", False)
                and need_dC
                and has_topk_repair
            ):
                raise RuntimeError(
                    "dWeight-only low-precision CCE backward is currently "
                    "validated only for the mixed MXFP8-row/MXFP8-dWeight "
                    "target-topk path"
                )
            repair_stream = None
            if need_dE and has_topk_repair and mxfp4_dE_stream is None:
                repair_stream = _launch_overlapped_topk_dE_repair(
                    dE,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            if getattr(ctx, "lowp_logits_bf16_dweight", False):
                if bf16_dweight_from_lowp_g.numel() != weight.numel():
                    raise RuntimeError(
                        "missing saved BF16 dWeight for the lowp-logit path"
                    )
                dC = bf16_dweight_from_lowp_g.mul(scale)
            elif ctx.g_mxfp8 or getattr(ctx, "g_mixed_dw_mxfp8_cols", False):
                dC = tk_mxfp8_gemm(
                    g_col,
                    x_col_q,
                    output_scale=(
                        scale
                        if getattr(ctx, "g_mixed_dw_mxfp8_cols", False)
                        else None
                    ),
                    config_env_name=(
                        "FP4_CCE_V4_MIXED_DW_MXFP8_GEMM_CONFIG"
                        if getattr(ctx, "g_mixed_dw_mxfp8_cols", False)
                        else "FP4_CCE_V4_MXFP8_GEMM_CONFIG"
                    ),
                )
                if not getattr(ctx, "g_mixed_dw_mxfp8_cols", False):
                    dC.mul_(scale)
            elif ctx.g_mxfp4:
                dC = (
                    tk_mxfp4_gemm_atbt(
                        g_row, x_col_q, output_scale=scale
                    )
                    if ctx.g_mxfp4_atbt_dw
                    else tk_mxfp4_gemm(g_col, x_col_q, output_scale=scale)
                )
            else:
                dC = tk_nvfp4_gemm(g_col, x_col_q)
            if mxfp4_dE_stream is not None:
                torch.cuda.current_stream(dC.device).wait_stream(
                    mxfp4_dE_stream
                )
                if has_topk_repair:
                    repair_stream = _launch_overlapped_topk_dE_repair(
                        dE,
                        weight,
                        targets,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        scale,
                        ctx.ignore_index,
                    )
            if compact_repair is not None:
                _finish_compact_topk_dC_repair(dC, compact_repair, scale)
            if getattr(ctx, "lowp_logits_bf16_dhidden", False):
                if (
                    compact_repair is None
                    and not getattr(ctx, "lowp_logits_bf16_dweight", False)
                ):
                    # Exact selected coefficients were included in the saved
                    # BF16 G before its dHidden GEMM.  Repair only dWeight.
                    sparse_correct_target_topk_dC(
                        dC,
                        x,
                        targets,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        scale,
                        ctx.ignore_index,
                    )
            elif not need_dE:
                if compact_repair is None:
                    # The outer LBT autograd bridge supplies a BF16 Cut-CCE
                    # dHidden. Preserve the existing low-precision dWeight,
                    # including its exact selected-token repair, without
                    # launching the discarded low-precision dHidden GEMM or
                    # repair kernel.
                    sparse_correct_target_topk_dC(
                        dC,
                        x,
                        targets,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        scale,
                        ctx.ignore_index,
                    )
            elif repair_stream is not None:
                if compact_repair is not None:
                    torch.cuda.current_stream(dC.device).wait_stream(
                        repair_stream
                    )
                else:
                    _finish_overlapped_topk_repair(
                        repair_stream,
                        dC,
                        x,
                        targets,
                        target_probs,
                        topk_indices,
                        topk_probs,
                        scale,
                        ctx.ignore_index,
                        (
                            prepared_vocab_rows,
                            prepared_x_rows,
                            prepared_coefficients,
                        )
                        if ctx.g_prepared_dC
                        else None,
                    )
            elif ctx.g_wide_topk_split:
                sparse_correct_target_topk_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top6_split:
                sparse_correct_target_top6_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top4_split:
                sparse_correct_target_top4_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top2_split:
                sparse_correct_target_top2_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_top1_split:
                sparse_correct_target_top1_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    topk_indices,
                    topk_probs,
                    scale,
                    ctx.ignore_index,
                )
            elif ctx.g_target_split:
                sparse_correct_target_split(
                    dE,
                    dC,
                    x,
                    weight,
                    targets,
                    target_probs,
                    scale,
                    ctx.ignore_index,
                )
            return (
                dE,
                None,
                None,
                None,
                None,
                None,
                None,
                dC,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        p_row = NVFP4Quantized(p_row_fp4, p_row_sc, p_row_sg * scale)
        w_col_q = NVFP4Quantized(w_col_fp4, w_col_sc, w_col_sg)

        dE = tk_nvfp4_gemm(p_row, w_col_q)
        if ctx.use_bf16_p_dweight:
            V = weight.shape[0]
            K = x.shape[1]
            dC = torch.empty(V, K, dtype=torch.bfloat16, device=x.device)
            chunk = _nvfp4_bf16_p_dweight_chunk()
            for start in range(0, V, chunk):
                end = min(start + chunk, V)
                dC[start:end] = probs[:, start:end].transpose(0, 1) @ x
            dC.mul_(scale)
        else:
            p_col = NVFP4Quantized(p_col_fp4, p_col_sc, p_col_sg * scale)
            x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
            dC = tk_nvfp4_gemm(p_col, x_col_q)

        if ctx.p_top4_split:
            sparse_correct_target_top4_split(
                dE,
                dC,
                x,
                weight,
                targets,
                target_probs,
                topk_indices,
                topk_probs,
                scale,
                ctx.ignore_index,
            )
        elif ctx.p_top1_split:
            sparse_correct_target_top1_split(
                dE,
                dC,
                x,
                weight,
                targets,
                target_probs,
                topk_indices,
                topk_probs,
                scale,
                ctx.ignore_index,
            )
        elif ctx.p_target_split:
            sparse_correct_target_split(
                dE,
                dC,
                x,
                weight,
                targets,
                target_probs,
                scale,
                ctx.ignore_index,
            )
        else:
            sparse_correct(dE, dC, x, weight, targets, scale, ctx.ignore_index)

        return (
            dE,
            None,
            None,
            None,
            None,
            None,
            None,
            dC,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class NVFP4CCE_VocabParallel_Function(torch.autograd.Function):
    """Bridge TP CCE over local vocab shards with FP4 local logits."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        targets,
        ignore_index,
        global_vocab_size,
        vocab_start,
        tp_group,
        reduce_dE,
        encode_centric,
    ):
        encode_centric = bool(encode_centric)
        if _use_nvfp4_group_input_quant() and encode_centric:
            q_x, q_x_col, q_w, q_w_col = quantize_both_nvfp4_row_and_col_tk(
                x, weight, encode_centric=True
            )
        else:
            quantize_x = _select_nvfp4_xw_quantizer("X")
            quantize_w = _select_nvfp4_xw_quantizer("W")
            q_x, q_x_col = quantize_x(x, encode_centric=encode_centric)
            q_w, q_w_col = quantize_w(weight, encode_centric=encode_centric)

        valid = targets.ne(int(ignore_index))
        if (
            use_nvfp4_vocab_parallel_direct_g_cache()
            and _use_nvfp4_vocab_parallel_chunked_logits_g_cache(
                int(q_x.fp4.shape[0]),
                int(q_w.fp4.shape[0]),
            )
        ):
            if _use_nvfp4_vocab_parallel_chunked_recompute():
                if not encode_centric or not _use_nvfp4_g_constant_scale():
                    raise RuntimeError(
                        "chunked vocab-parallel NVFP4 recompute requires encode-centric constant-scale G"
                    )
                loss, global_lse = _nvfp4_vocab_parallel_chunked_loss_lse(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(global_vocab_size),
                    int(vocab_start),
                    tp_group,
                )
                ctx.save_for_backward(
                    q_x.fp4,
                    q_x.sc,
                    q_x.sg,
                    q_x_col.fp4,
                    q_x_col.sc,
                    q_x_col.sg,
                    q_w.fp4,
                    q_w.sc,
                    q_w.sg,
                    q_w_col.fp4,
                    q_w_col.sc,
                    q_w_col.sg,
                    targets,
                    global_lse,
                )
                ctx.ignore_index = int(ignore_index)
                ctx.tp_group = tp_group
                ctx.reduce_dE = bool(reduce_dE)
                ctx.global_vocab_size = int(global_vocab_size)
                ctx.vocab_start = int(vocab_start)
                ctx.local_valid_cols = max(
                    min(int(global_vocab_size) - int(vocab_start), int(q_w.fp4.shape[0])),
                    0,
                )
                ctx.encode_centric = encode_centric
                ctx.use_chunked_recompute = True
                return loss
            (
                loss,
                g_row_fp4,
                g_row_sc,
                g_row_sg,
                g_col_fp4,
                g_col_sc,
                g_col_sg,
            ) = _nvfp4_vocab_parallel_chunked_logits_g_cache(
                q_x,
                q_w,
                targets,
                int(ignore_index),
                int(global_vocab_size),
                int(vocab_start),
                tp_group,
                encode_centric=encode_centric,
            )
        else:
            logits = tk_nvfp4_gemm(q_x, q_w)
            if use_nvfp4_vocab_parallel_direct_g_cache():
                (
                    loss,
                    g_row_fp4,
                    g_row_sc,
                    g_row_sg,
                    g_col_fp4,
                    g_col_sc,
                    g_col_sg,
                ) = nvfp4_vocab_parallel_tiled_g_cache(
                    logits,
                    targets,
                    valid,
                    int(vocab_start),
                    int(global_vocab_size),
                    tp_group,
                )
            else:
                loss, grad_probs = vocab_parallel_loss_and_grad_probs(
                    logits,
                    targets,
                    valid,
                    int(vocab_start),
                    int(global_vocab_size),
                    tp_group,
                )
                g_row_q, g_col_q = _select_nvfp4_g_quantizer()(
                    grad_probs,
                    encode_centric=encode_centric,
                )
                g_row_fp4, g_row_sc, g_row_sg = g_row_q.fp4, g_row_q.sc, g_row_q.sg
                g_col_fp4, g_col_sc, g_col_sg = g_col_q.fp4, g_col_q.sc, g_col_q.sg
        ctx.use_chunked_recompute = False
        ctx.save_for_backward(
            g_row_fp4,
            g_row_sc,
            g_row_sg,
            g_col_fp4,
            g_col_sc,
            g_col_sg,
            q_x_col.fp4,
            q_x_col.sc,
            q_x_col.sg,
            q_w_col.fp4,
            q_w_col.sc,
            q_w_col.sg,
            targets,
        )
        ctx.ignore_index = int(ignore_index)
        ctx.tp_group = tp_group
        ctx.reduce_dE = bool(reduce_dE)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, "use_chunked_recompute", False):
            return _nvfp4_vocab_parallel_chunked_recompute_backward(ctx, grad_output)

        (
            g_row_fp4,
            g_row_sc,
            g_row_sg,
            g_col_fp4,
            g_col_sc,
            g_col_sg,
            x_col_fp4,
            x_col_sc,
            x_col_sg,
            w_col_fp4,
            w_col_sc,
            w_col_sg,
            targets,
        ) = ctx.saved_tensors

        n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
        scale = (grad_output.float() / n_valid).reshape(())
        g_row = NVFP4Quantized(g_row_fp4, g_row_sc, g_row_sg * scale)
        g_col = NVFP4Quantized(g_col_fp4, g_col_sc, g_col_sg * scale)
        x_col_q = NVFP4Quantized(x_col_fp4, x_col_sc, x_col_sg)
        w_col_q = NVFP4Quantized(w_col_fp4, w_col_sc, w_col_sg)

        # Return this rank's vocab-shard contribution. When Bridge feeds a
        # replicated hidden state, overlap the required TP input-gradient
        # reduction with the independent weight-gradient GEMM.
        dE = tk_nvfp4_gemm(g_row, w_col_q)
        dE_reduce = None
        if ctx.reduce_dE:
            dE_reduce = torch.distributed.all_reduce(dE, group=ctx.tp_group, async_op=True)
        dC = tk_nvfp4_gemm(g_col, x_col_q)
        if dE_reduce is not None:
            dE_reduce.wait()
        return dE, dC, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def nvfp4_cce_tk(
    x: torch.Tensor,       # (N, D) BF16 embeddings
    weight: torch.Tensor,   # (V, D) BF16 classifier weights
    targets: torch.Tensor,  # (N,) int64
    ignore_index: int = -100,
) -> torch.Tensor:
    """NVFP4 Cross-Entropy using ThunderKittens GEMM.

    Quantizes both input and weight to NVFP4, computes logits via TK GEMM,
    then applies F.cross_entropy.
    """
    # Pad M and V to multiples of 256
    M, K = x.shape
    V = weight.shape[0]
    ALIGN = 256

    M_pad = ((M + ALIGN - 1) // ALIGN) * ALIGN
    V_pad = ((V + ALIGN - 1) // ALIGN) * ALIGN

    if M_pad != M:
        x = F.pad(x, (0, 0, 0, M_pad - M))
        targets = F.pad(targets, (0, M_pad - M), value=ignore_index)
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    q_x = quantize_nvfp4_tk(x)
    q_w = quantize_nvfp4_tk(weight)

    return NVFP4CCE_TK_Function.apply(
        q_x.fp4, q_x.sc, q_x.sg,
        q_w.fp4, q_w.sc, q_w.sg,
        weight, x,
        targets.to(torch.int64), ignore_index,
    )


def nvfp4_cce_tk_v4_pcache(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    encode_centric: bool = True,
) -> torch.Tensor:
    """NVFP4 CCE v4 prototype using a quantized softmax-probability cache."""
    M, _K = x.shape
    V = weight.shape[0]
    ALIGN = 256

    M_pad = ((M + ALIGN - 1) // ALIGN) * ALIGN
    V_pad = ((V + ALIGN - 1) // ALIGN) * ALIGN

    if M_pad != M:
        x = F.pad(x, (0, 0, 0, M_pad - M))
        targets = F.pad(targets, (0, M_pad - M), value=ignore_index)
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    return NVFP4CCE_PCache_Function.apply(
        x,
        weight,
        targets.to(torch.int64),
        int(ignore_index),
        int(V),
        bool(encode_centric),
    )


def nvfp4_cce_tk_v4_pcache_prequantized_x(
    x: torch.Tensor,
    x_q: NVFP4Quantized | MXFP8Quantized | MXFP6Quantized | MXFP4Quantized | DirectFP8Quantized,
    x_col_q: NVFP4Quantized | MXFP8Quantized | MXFP4Quantized,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    encode_centric: bool = True,
    weight_quantized: tuple | None = None,
) -> torch.Tensor:
    """NVFP4 CCE v4 with forward/column operands supplied by the producer.

    `x` is still required for the exact sparse label correction and for the
    gradient returned to the producer. `x_q` may be an NVFP4, MXFP8, MXFP6,
    or MXFP4 forward operand. The backward column may use MXFP8, MXFP4, or
    NVFP4 and
    must match the selected G-cache mode. The quantized tensors must correspond
    to the same (possibly padded) `x`. `weight_quantized`, when supplied, must
    contain row and column operands produced from the current `weight` value.
    """
    M, _K = x.shape
    V = weight.shape[0]
    ALIGN = 256

    M_pad = ((M + ALIGN - 1) // ALIGN) * ALIGN
    V_pad = ((V + ALIGN - 1) // ALIGN) * ALIGN
    if M_pad != M:
        raise ValueError("prequantized x path requires caller-padded M to a multiple of 256")
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    use_mxfp8_forward = isinstance(x_q, MXFP8Quantized)
    use_mxfp6_forward = isinstance(x_q, MXFP6Quantized)
    use_mxfp4_forward = isinstance(x_q, MXFP4Quantized)
    use_direct_fp8_forward = isinstance(x_q, DirectFP8Quantized)
    use_mxfp8_col = isinstance(x_col_q, MXFP8Quantized)
    use_mxfp4_col = isinstance(x_col_q, MXFP4Quantized)
    use_mixed_dw_mxfp8_cols = _use_mixed_dw_mxfp8_cols()
    if use_mxfp8_col and use_mxfp4_col:
        raise ValueError("x column operand cannot be both MXFP8 and MXFP4")
    if _use_mxfp8_g_cache() and not (use_mxfp8_forward and use_mxfp8_col):
        raise ValueError("MXFP8 G-cache requires MXFP8 row and column x operands")
    if _use_mxfp4_g_cache() and not use_mxfp4_col:
        raise ValueError("MXFP4 G-cache requires an MXFP4 column x operand")
    if _use_mxfp8_row_nvfp4_col_g_cache():
        if use_mixed_dw_mxfp8_cols:
            if not (use_mxfp8_forward and use_mxfp8_col):
                raise ValueError(
                    "mixed G-cache MXFP8 dWeight mode requires MXFP8 row and "
                    "column x operands"
                )
        elif not (
            (use_mxfp4_forward or use_mxfp8_forward)
            and not use_mxfp8_col
            and not use_mxfp4_col
        ):
            raise ValueError(
                "mixed G-cache requires an MXFP4 or MXFP8 row and NVFP4 "
                "column x operand"
            )
    elif use_mixed_dw_mxfp8_cols:
        raise ValueError(
            "FP4_CCE_V4_MIXED_DW_MXFP8_COLS requires the mixed G-cache"
        )
    if use_mxfp8_col and not (
        _use_mxfp8_g_cache() or use_mixed_dw_mxfp8_cols
    ):
        raise ValueError(
            "MXFP8 column x operand requires the MXFP8 G-cache or mixed "
            "MXFP8 dWeight columns"
        )
    if use_mxfp4_col and not _use_mxfp4_g_cache():
        raise ValueError("MXFP4 column x operand requires FP4_CCE_V4_MXFP4_G_CACHE=1")
    x_forward = (
        x_q.fp6
        if use_mxfp6_forward
        else x_q.fp8
        if use_mxfp8_forward or use_direct_fp8_forward
        else x_q.fp4
    )
    x_col_data = x_col_q.fp8 if use_mxfp8_col else x_col_q.fp4
    if x_forward.shape[0] != x.shape[0]:
        raise ValueError("prequantized x tensors do not match x shape")
    column_omitted = x_col_data.numel() == 0 or x_col_q.sc.numel() == 0
    if column_omitted:
        if not (
            _use_lowp_logits_bf16_both_inplace_g()
            and use_mxfp8_forward
            and use_mxfp8_col
            and x_col_data.numel() == 0
            and x_col_q.sc.numel() == 0
        ):
            raise ValueError(
                "empty x column operand is valid only for exact BF16-both "
                "cache elision"
            )
    elif x_col_data.shape[0] != x.shape[1]:
        raise ValueError("prequantized x tensors do not match x shape")

    x_sg = (
        torch.empty(0, dtype=torch.float32, device=x.device)
        if (
            use_mxfp8_forward
            or use_mxfp6_forward
            or use_mxfp4_forward
            or use_direct_fp8_forward
        )
        else x_q.sg
    )
    x_col_sg = (
        torch.empty(0, dtype=torch.float32, device=x.device)
        if use_mxfp8_col or use_mxfp4_col
        else x_col_q.sg
    )

    return NVFP4CCE_PCache_PrequantX_Function.apply(
        x,
        x_forward,
        x_q.sc,
        x_sg,
        x_col_data,
        x_col_q.sc,
        x_col_sg,
        weight,
        targets.to(torch.int64),
        int(ignore_index),
        int(V),
        bool(encode_centric),
        3
        if use_direct_fp8_forward
        else (4 if x_q.format == "e2m3" else 5)
        if use_mxfp6_forward
        else 1
        if use_mxfp8_forward
        else 2
        if use_mxfp4_forward
        else 0,
        weight_quantized,
    )


def nvfp4_cce_tk_v4_vocab_parallel(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    global_vocab_size: int | None = None,
    vocab_start: int = 0,
    tp_group=None,
    reduce_dE: bool = False,
    encode_centric: bool = True,
) -> torch.Tensor:
    """NVFP4 v4 CCE for Bridge vocab-parallel output weights."""
    M, _K = x.shape
    V = weight.shape[0]
    M_ALIGN = 128
    V_ALIGN = 256

    M_pad = ((M + M_ALIGN - 1) // M_ALIGN) * M_ALIGN
    V_pad = ((V + V_ALIGN - 1) // V_ALIGN) * V_ALIGN

    if M_pad != M:
        x = F.pad(x, (0, 0, 0, M_pad - M))
        targets = F.pad(targets, (0, M_pad - M), value=ignore_index)
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    if global_vocab_size is None:
        global_vocab_size = int(vocab_start) + int(V)

    return NVFP4CCE_VocabParallel_Function.apply(
        x,
        weight,
        targets.to(torch.int64),
        int(ignore_index),
        int(global_vocab_size),
        int(vocab_start),
        tp_group,
        bool(reduce_dE),
        bool(encode_centric),
    )
