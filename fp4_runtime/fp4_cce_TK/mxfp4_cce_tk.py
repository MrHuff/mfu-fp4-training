"""
MXFP4 Cross-Entropy using ThunderKittens GEMM.

Replaces CUTLASS (qutlass) MXFP4 GEMM with TK's mxfp4_gemm kernel.
Quantization also uses TK's mxfp4_quantize kernel.

D(M, N) = A(M, K) @ B(N, K)^T  (TN layout)
"""

import os
import sys
import importlib.util
import torch
import torch.nn.functional as F
import torch.distributed as dist

from fp4_cce_TK.extension_loader import find_existing_extension

from fp4_cce_TK.v4_common import (
    assume_all_valid_full_vocab,
    backward_scale_cuda,
    bf16_logits_cuda,
    bf16_tail_grads_cuda,
    direct_loss_and_probs,
    direct_loss_and_probs_target_split,
    direct_loss_and_probs_target_top1_split,
    direct_loss_and_probs_target_top4_split,
    direct_loss_and_grad_probs,
    loss_and_probs,
    mxfp4_direct_p_cache,
    mxfp4_staged_g_cache,
    mxfp4_tiled_g_cache,
    mxfp4_vocab_parallel_tiled_g_cache,
    sparse_correct,
    sparse_correct_target_split,
    sparse_correct_target_top1_split,
    sparse_correct_target_top4_split,
    sparse_correct_scaled_dE,
    use_direct_mxfp4_p_cache,
    use_mxfp4_vocab_parallel_direct_g_cache,
    valid_mask_count_cuda,
    vocab_parallel_loss_and_grad_probs,
)

# TK quantizer uses E8M0 = round(log2(amax))+127, encoding the raw amax.
# This means FP4 values are scaled by 6/amax, and dequant gives val*amax.
# Since FP4 max is 6, each element is effectively multiplied by 6 after
# dequant. With two operands, the GEMM output is 6² = 36x too large.
MXFP4_ALPHA = 1.0 / 36.0


def _mxfp4_v4_p_cache_mode() -> str:
    return os.environ.get("FP4_CCE_V4_DIRECT_MXFP4_P_CACHE_MODE", "auto").strip().lower()


def _use_mxfp4_p_target_split() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_P_TARGET_SPLIT", "0") != "0"


def _mxfp4_p_topk_split() -> int:
    raw = os.environ.get("FP4_CCE_V4_MXFP4_P_TOPK_SPLIT", "0")
    topk = int(raw)
    if topk not in {0, 1, 4}:
        raise ValueError("FP4_CCE_V4_MXFP4_P_TOPK_SPLIT must be 0, 1, or 4")
    return topk


def _mxfp4_auto_inductor_elements() -> int:
    return int(os.environ.get("FP4_CCE_V4_MXFP4_AUTO_INDUCTOR_ELEMENTS", str(4096 * 32768)))


def _use_mxfp4_direct_loss_probs_fallback() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_DIRECT_LOSS_PROBS_FALLBACK", "1") != "0"


def _use_mxfp4_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_G_CACHE", "1") != "0"


def _use_mxfp4_g_target_split() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_G_TARGET_SPLIT", "0") != "0"


def _use_mxfp4_fused_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_FUSED_G_CACHE", "0") != "0"


def _use_mxfp4_staged_g_cache() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_STAGED_G_CACHE", "0") != "0"


def _mxfp4_fused_g_cache_impl() -> str:
    return os.environ.get("FP4_CCE_V4_MXFP4_FUSED_G_CACHE_IMPL", "direct").strip().lower()


def _use_mxfp4_direct_g_producer() -> bool:
    return os.environ.get("FP4_CCE_V4_MXFP4_DIRECT_G_PRODUCER", "1") != "0"


def _use_mxfp4_chunked_recompute() -> bool:
    return os.environ.get(
        "FP4_CCE_V4_MXFP4_CHUNKED_RECOMPUTE",
        "0",
    ).strip().lower() not in {"0", "false", "no", "off"}


def _mxfp4_loss_and_probs(logits, targets, valid, ignore_index, vocab_size):
    if _use_mxfp4_direct_loss_probs_fallback():
        return direct_loss_and_probs(logits, targets, valid, int(vocab_size))
    return loss_and_probs(logits, targets, valid, int(ignore_index))

# ---------------------------------------------------------------------------
# Lazy TK MXFP4 GEMM import
# ---------------------------------------------------------------------------
_tk_mxfp4 = None

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
    ])
    deduped = []
    for root in roots:
        if root and root not in deduped:
            deduped.append(root)
    return deduped


def _find_existing_so(label, relpath):
    return find_existing_extension(label, _repo_roots(), relpath)


def _get_tk_mxfp4():
    global _tk_mxfp4
    if _tk_mxfp4 is not None:
        return _tk_mxfp4

    so_name = '_C_mx.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "TK MXFP4 _C_mx.so",
        os.path.join('ThunderKittens', 'kernels', 'gemm', 'mxfp4_gb200', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    old_c = sys.modules.pop('_C_mx', None)
    spec = importlib.util.spec_from_file_location('_C_mx', so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_C_mx'] = mod
    spec.loader.exec_module(mod)
    sys.modules.pop('_C_mx', None)

    _tk_mxfp4 = mod
    return _tk_mxfp4


# ---------------------------------------------------------------------------
# Lazy MXFP4 v3 quantizer import (pipelined kernel, 3x faster than v2)
# ---------------------------------------------------------------------------
import ctypes

_mxfp4_quant = None
_mxfp4_quant_v4 = None

def _get_mxfp4_quant():
    global _mxfp4_quant
    if _mxfp4_quant is not None:
        return _mxfp4_quant

    so_name = 'mxfp4_quant_v3.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "mxfp4_quant_v3",
        os.path.join('TK_quantisation', 'mxfp4_v3', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    # Need torch libs for loading
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_python.so'), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location('mxfp4_quant_v3', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _mxfp4_quant = mod
    return _mxfp4_quant


def _get_mxfp4_quant_v4():
    global _mxfp4_quant_v4
    if _mxfp4_quant_v4 is not None:
        return _mxfp4_quant_v4

    so_name = 'mxfp4_quant_v4.cpython-312-aarch64-linux-gnu.so'
    so_path = _find_existing_so(
        "mxfp4_quant_v4",
        os.path.join('TK_quantisation', 'mxfp4_v4', so_name),
    )

    if not torch.cuda.is_initialized():
        torch.cuda.init()
        _ = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()

    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    ctypes.CDLL(os.path.join(torch_lib, 'libtorch_python.so'), mode=ctypes.RTLD_GLOBAL)

    spec = importlib.util.spec_from_file_location('mxfp4_quant_v4', so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    _mxfp4_quant_v4 = mod
    return _mxfp4_quant_v4


# ---------------------------------------------------------------------------
# MXFP4 Quantization via TK
# ---------------------------------------------------------------------------
class MXFP4Quantized:
    __slots__ = ['fp4', 'sc', 'bf16']

    def __init__(self, fp4, sc, bf16=None):
        self.fp4 = fp4    # [M, K//2] fp4x2
        self.sc = sc      # [M//128, K//128, 32, 16] uint8 (E8M0 scales)
        self.bf16 = bf16  # original bf16 data (for backward)


def quantize_mxfp4_tk(x: torch.Tensor, keep_bf16: bool = True,
                      mode: int = 1) -> MXFP4Quantized:
    """Quantize BF16 (M, K) → MXFP4.
    mode: 0=RTE, 1=ENCODE (default, best accuracy), 2=DECODE.
    """
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    quant = _get_mxfp4_quant()
    fp4, sc = quant.mxfp4_quantize_for_gemm(x, mode)

    return MXFP4Quantized(fp4, sc, bf16=x if keep_bf16 else None)


def quantize_mxfp4_row_tk(
    x: torch.Tensor, mode: int = 1
) -> MXFP4Quantized:
    """Produce only the MXFP4 row operand used by a forward GEMM."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("MXFP4 row input must be contiguous BF16 [M, K]")
    M, K = x.shape
    if M % 128 or K % 128:
        raise ValueError(
            f"MXFP4 dimensions must be multiples of 128, got M={M}, K={K}"
        )

    row_fp4, row_sc = _get_mxfp4_quant_v4().mxfp4_quantize_for_gemm(
        x, int(mode)
    )
    return MXFP4Quantized(row_fp4, row_sc)


def quantize_mxfp4_row_and_col_tk(
    x: torch.Tensor, mode: int = 1, *, role: str | None = None
) -> tuple:
    """Quantize BF16 (M, K) → MXFP4 for BOTH row and col (transposed) in one call.

    Returns: (row_q: MXFP4Quantized, col_q: MXFP4Quantized)
    where col_q is the MXFP4 quantization of x^T.
    """
    if role not in {None, "X", "W"}:
        raise ValueError(f"MXFP4 head quantization role must be X or W, got {role!r}")
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0

    data_sr = os.environ.get("FP4_CCE_V4_MXFP4_FORWARD_DATA_SR", "0") != "0"
    scale_sr = os.environ.get("FP4_CCE_V4_MXFP4_FORWARD_SCALE_SR", "0") != "0"
    row_rht = os.environ.get("FP4_CCE_V4_MXFP4_FORWARD_ROW_RHT", "0") != "0"
    rht_block_size = int(
        os.environ.get("FP4_CCE_V4_MXFP4_FORWARD_RHT_BLOCK_SIZE", "16")
    )
    if row_rht and rht_block_size not in {16, 32}:
        raise ValueError(
            "MXFP4 forward row RHT block size must be 16 or 32, got "
            f"{rht_block_size}"
        )
    if role is not None:
        role_data_sr = os.environ.get(
            f"FP4_CCE_V4_MXFP4_FORWARD_{role}_DATA_SR"
        )
        role_scale_sr = os.environ.get(
            f"FP4_CCE_V4_MXFP4_FORWARD_{role}_SCALE_SR"
        )
        if role_data_sr is not None:
            data_sr = role_data_sr != "0"
        if role_scale_sr is not None:
            scale_sr = role_scale_sr != "0"

    quant = _get_mxfp4_quant_v4()
    if data_sr or scale_sr or row_rht:
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
            os.environ.get(
                "FP4_CCE_V4_MXFP4_FORWARD_RNG_SUBSEQUENCE_BASE", "0"
            )
        )
        quantize = (
            quant.mxfp4_quantize_row_and_col_opt_rht_row_only
            if row_rht
            else quant.mxfp4_quantize_row_and_col_opt
        )
        args = [
            x,
            int(mode),
            data_sr,
            scale_sr,
        ]
        if row_rht:
            args.extend((rht_block_size, False))
        args.extend((rng_seed, rng_subsequence))
        row_fp4, row_sc, col_fp4, col_sc = (
            quantize(
                *args,
            )
        )
    else:
        row_fp4, row_sc, col_fp4, col_sc = (
            quant.mxfp4_quantize_row_and_col(x, int(mode))
        )

    return (MXFP4Quantized(row_fp4, row_sc),
            MXFP4Quantized(col_fp4, col_sc))


def quantize_mxfp4_col_tk(
    x: torch.Tensor, mode: int = 1
) -> MXFP4Quantized:
    """Quantize only the transposed MXFP4 operand used by backward GEMMs."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("MXFP4 column input must be contiguous BF16 [M, K]")
    M, K = x.shape
    if M % 128 or K % 128:
        raise ValueError(
            f"MXFP4 dimensions must be multiples of 128, got M={M}, K={K}"
        )

    col_fp4, col_sc = _get_mxfp4_quant_v4().mxfp4_quantize_col_only(
        x, int(mode)
    )
    return MXFP4Quantized(col_fp4, col_sc)


def quantize_mxfp4_norm_row_and_col_tk(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    mode: int = 1,
):
    """Fuse RMSNorm with MXFP4 row/col quantization for producer-side CCE x.

    The fused kernel intentionally returns only the quantized contracts plus
    inv_rms. The caller still supplies materialized BF16 RMSNorm output to CCE
    for exact sparse label correction and autograd into the producer.
    """
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    assert gamma.ndim == 1 and gamma.dtype == torch.bfloat16
    assert x.shape[1] == gamma.shape[0]
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    quant = _get_mxfp4_quant_v4()
    row_fp4, row_sc, col_fp4, col_sc, inv_rms = quant.mxfp4_fused_rmsnorm_quantize_row_and_col(
        x,
        gamma,
        float(epsilon),
        int(mode),
    )
    return (
        MXFP4Quantized(row_fp4, row_sc),
        MXFP4Quantized(col_fp4, col_sc),
        inv_rms,
    )


def quantize_mxfp4_norm_row_and_col_with_output_tk(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-5,
    mode: int = 1,
):
    """Fuse RMSNorm, BF16 output materialization, and MXFP4 row/col quantization."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16
    assert gamma.ndim == 1 and gamma.dtype == torch.bfloat16
    assert x.shape[1] == gamma.shape[0]
    M, K = x.shape
    assert K % 128 == 0 and M % 128 == 0, f"Dims must be multiples of 128, got M={M}, K={K}"

    quant = _get_mxfp4_quant_v4()
    fused_with_output = getattr(quant, "mxfp4_fused_rmsnorm_quantize_row_and_col_with_output", None)
    if fused_with_output is not None:
        normed, row_fp4, row_sc, col_fp4, col_sc, inv_rms = (
            fused_with_output(
                x,
                gamma,
                float(epsilon),
                int(mode),
            )
        )
    else:
        row_fp4, row_sc, col_fp4, col_sc, inv_rms = (
            quant.mxfp4_fused_rmsnorm_quantize_row_and_col(
                x,
                gamma,
                float(epsilon),
                int(mode),
            )
        )
        normed = (x.float() * inv_rms.unsqueeze(1) * gamma).to(torch.bfloat16)
    return (
        normed,
        MXFP4Quantized(row_fp4, row_sc),
        MXFP4Quantized(col_fp4, col_sc),
        inv_rms,
    )


def quantize_both_mxfp4_tk(x: torch.Tensor, w: torch.Tensor,
                           keep_bf16: bool = True):
    """Quantize both x(M,K) and w(V,K) in a single grouped kernel launch.

    Returns (x_q: MXFP4Quantized, w_q: MXFP4Quantized).
    """
    assert x.ndim == 2 and w.ndim == 2 and x.dtype == torch.bfloat16
    M, K = x.shape
    V = w.shape[0]
    assert w.shape[1] == K
    assert K % 128 == 0 and M % 128 == 0 and V % 128 == 0

    # Stack vertically and quantize in one launch
    stacked = torch.cat([x, w], dim=0)  # (M+V, K)
    quant = _get_mxfp4_quant()
    results = quant.mxfp4_group_quantize_dim0(stacked, [M, V])

    # Each result is (fp4, scales) tuple
    x_fp4, x_sc = results[0]
    w_fp4, w_sc = results[1]

    x_q = MXFP4Quantized(x_fp4, x_sc, bf16=x if keep_bf16 else None)
    w_q = MXFP4Quantized(w_fp4, w_sc, bf16=w if keep_bf16 else None)
    return x_q, w_q


def _mxfp4_gemm_config(M: int, N: int, K: int):
    """Select a measured dense-GEMM config without changing other shapes."""
    raw = os.environ.get("FP4_CCE_V4_MXFP4_GEMM_CONFIG", "auto").strip().lower()
    if raw in {"", "auto"}:
        # GB200 sweep: config 16 overlaps the four-stage epilogue and is
        # bit-identical while substantially faster for the Llama-8B CCE
        # logits GEMM. Other shapes retain the generic config.
        return 16 if (M, N, K) == (4096, 128256, 2048) else None
    if raw in {"default", "none", "off", "-1"}:
        return None
    config = int(raw)
    if config not in range(23):
        raise ValueError("FP4_CCE_V4_MXFP4_GEMM_CONFIG must be auto, default, or 0-22")
    return config


def tk_mxfp4_gemm(A_q: MXFP4Quantized, B_q: MXFP4Quantized,
                  out: torch.Tensor = None,
                  output_scale: torch.Tensor = None) -> torch.Tensor:
    """D(M, N) = A @ B^T using TK MXFP4 GEMM."""
    tk = _get_tk_mxfp4()
    M = A_q.fp4.shape[0]
    N = B_q.fp4.shape[0]
    use_empty = os.environ.get(
        "FP4_CCE_V4_MXFP4_EMPTY_GEMM_OUTPUT",
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
    if output_scale is None:
        K = A_q.fp4.shape[1] * 2
        config = _mxfp4_gemm_config(M, N, K)
        if config is not None and hasattr(tk, "mxfp4_gemm_config"):
            tk.mxfp4_gemm_config(
                A_q.fp4, A_q.sc, B_q.fp4, B_q.sc, out, config
            )
        else:
            tk.mxfp4_gemm(A_q.fp4, A_q.sc, B_q.fp4, B_q.sc, out)
    elif hasattr(tk, "mxfp4_gemm_scaled"):
        tk.mxfp4_gemm_scaled(A_q.fp4, A_q.sc, B_q.fp4, B_q.sc, out, output_scale.reshape(1))
    else:
        tk.mxfp4_gemm(A_q.fp4, A_q.sc, B_q.fp4, B_q.sc, out)
        out.mul_(output_scale)
    # Alpha (1/36) is now fused in the GEMM kernel epilogue
    return out


def tk_mxfp4_gemm_atbt(
    a_row_q: MXFP4Quantized,
    b_col_q: MXFP4Quantized,
    out: torch.Tensor = None,
    output_scale: torch.Tensor = None,
) -> torch.Tensor:
    """Compute ``a_row_q.T @ b_col_q.T`` without materializing A columns.

    ``a_row_q`` stores logical ``[K, M]`` values in row orientation while
    ``b_col_q`` stores logical ``[N, K]`` values in transposed orientation.
    This is the layout pair produced by the MXFP4 G-row and X-column caches.
    """
    tk = _get_tk_mxfp4()
    if not hasattr(tk, "mxfp4_gemm_atbt"):
        raise RuntimeError("TK MXFP4 extension does not provide mxfp4_gemm_atbt")
    k = a_row_q.fp4.shape[0]
    m = a_row_q.fp4.shape[1] * 2
    n = b_col_q.fp4.shape[0]
    if b_col_q.fp4.shape[1] * 2 != k:
        raise ValueError("MXFP4 AtBt operands have incompatible K dimensions")
    if out is None:
        out = torch.empty(m, n, dtype=torch.bfloat16, device=a_row_q.fp4.device)
    tk.mxfp4_gemm_atbt(
        a_row_q.fp4,
        a_row_q.sc,
        b_col_q.fp4,
        b_col_q.sc,
        out,
    )
    if output_scale is not None:
        out.mul_(output_scale)
    return out


def _mxfp4_chunked_logits_chunk() -> int:
    raw = os.environ.get(
        "FP4_CCE_V4_MXFP4_CHUNKED_LOGITS_G_CACHE_CHUNK",
        os.environ.get("FP4_CCE_V4_MXFP4_CHUNKED_LOGITS_CHUNK", "2048"),
    )
    return max(int(raw), 128)


def _use_mxfp4_chunked_logits_g_cache(M: int, V: int) -> bool:
    raw = os.environ.get("FP4_CCE_V4_MXFP4_CHUNKED_LOGITS_G_CACHE", "0").strip().lower()
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


def _mxfp4_vocab_parallel_chunked_logits_g_cache(
    q_x: MXFP4Quantized,
    q_w: MXFP4Quantized,
    targets: torch.Tensor,
    ignore_index: int,
    global_vocab_size: int,
    vocab_start: int,
    tp_group,
    mode: int = 1,
):
    M = int(q_x.fp4.shape[0])
    V = int(q_w.fp4.shape[0])
    device = q_x.fp4.device
    chunk = _mxfp4_chunked_logits_chunk()
    chunk = ((chunk + 127) // 128) * 128
    chunk = max(128, min(chunk, V))

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
        q_w_chunk = MXFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
        )
        logits = tk_mxfp4_gemm(q_x, q_w_chunk)
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
        q_w_chunk = MXFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
        )
        logits = tk_mxfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        logits_f.sub_(global_max[:, None])
        logits_f.exp_()
        local_den.add_(logits_f.sum(dim=1))
        del logits, logits_f

    global_den = local_den.clone()
    if use_dist:
        dist.all_reduce(global_den, op=dist.ReduceOp.SUM, group=tp_group)
    global_lse = global_max + torch.log(global_den)
    denom = valid.sum().clamp(min=1)
    row_loss = torch.where(valid, global_lse - target_logits, torch.zeros_like(global_lse))
    loss = row_loss.sum() / denom

    row_fp4 = torch.empty((M, V // 2), dtype=torch.float4_e2m1fn_x2, device=device)
    row_sc = torch.empty((M // 128, V // 128, 32, 16), dtype=torch.uint8, device=device)
    col_fp4 = torch.empty((V, M // 2), dtype=torch.float4_e2m1fn_x2, device=device)
    col_sc = torch.empty((V // 128, M // 128, 32, 16), dtype=torch.uint8, device=device)
    row_fp4_u8 = row_fp4.view(torch.uint8)
    col_fp4_u8 = col_fp4.view(torch.uint8)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = MXFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
        )
        logits = tk_mxfp4_gemm(q_x, q_w_chunk)
        logits_f = logits.float()
        if local_valid_cols < end:
            valid_cols = max(local_valid_cols - start, 0)
            logits_f[:, valid_cols:] = -float("inf")
        logits_f.sub_(global_lse[:, None])
        logits_f.exp_()
        grad = logits_f
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
        g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad.to(torch.bfloat16), mode=mode)
        row_fp4_u8[:, start // 2 : end // 2].copy_(g_row_q.fp4.view(torch.uint8))
        row_sc[:, start // 128 : end // 128].copy_(g_row_q.sc)
        col_fp4_u8[start:end].copy_(g_col_q.fp4.view(torch.uint8))
        col_sc[start // 128 : end // 128].copy_(g_col_q.sc)
        del logits, logits_f, grad, g_row_q, g_col_q

    return loss, row_fp4, row_sc, col_fp4, col_sc


def _mxfp4_vocab_parallel_chunked_loss_lse(
    q_x: MXFP4Quantized,
    q_w: MXFP4Quantized,
    targets: torch.Tensor,
    ignore_index: int,
    global_vocab_size: int,
    vocab_start: int,
    tp_group,
):
    M = int(q_x.fp4.shape[0])
    V = int(q_w.fp4.shape[0])
    device = q_x.fp4.device
    chunk = _mxfp4_chunked_logits_chunk()
    chunk = ((chunk + 127) // 128) * 128
    chunk = max(128, min(chunk, V))

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
        q_w_chunk = MXFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
        )
        logits = tk_mxfp4_gemm(q_x, q_w_chunk)
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
        q_w_chunk = MXFP4Quantized(
            q_w.fp4[start:end],
            q_w.sc[start // 128 : end // 128],
        )
        logits = tk_mxfp4_gemm(q_x, q_w_chunk)
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


def _mxfp4_chunked_recompute_backward(ctx, grad_output):
    (
        x_row_fp4,
        x_row_sc,
        x_col_fp4,
        x_col_sc,
        w_row_fp4,
        w_row_sc,
        w_col_fp4,
        w_col_sc,
        targets,
        global_lse,
    ) = ctx.saved_tensors

    n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
    scale = (grad_output.float() / n_valid).reshape(())
    q_x = MXFP4Quantized(x_row_fp4, x_row_sc)
    x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)

    M = int(x_row_fp4.shape[0])
    V = int(w_row_fp4.shape[0])
    K = int(x_col_fp4.shape[0])
    device = x_row_fp4.device
    chunk = _mxfp4_chunked_logits_chunk()
    chunk = ((chunk + 127) // 128) * 128
    chunk = max(128, min(chunk, V))
    local_targets = targets.to(torch.long) - int(ctx.vocab_start)
    valid = targets.ne(ctx.ignore_index)
    local_valid_cols = int(ctx.local_valid_cols)

    dE = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
    dC = torch.empty((V, K), dtype=torch.bfloat16, device=device)
    w_col_fp4_u8 = w_col_fp4.view(torch.uint8)

    for start in range(0, V, chunk):
        end = min(start + chunk, V)
        q_w_chunk = MXFP4Quantized(
            w_row_fp4[start:end],
            w_row_sc[start // 128 : end // 128],
        )
        logits = tk_mxfp4_gemm(q_x, q_w_chunk)
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
        g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_bf16, mode=ctx.mode)
        del grad_bf16

        w_col_fp4_chunk = (
            w_col_fp4_u8[:, start // 2 : end // 2]
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
        )
        w_col_chunk = MXFP4Quantized(
            w_col_fp4_chunk,
            w_col_sc[:, start // 128 : end // 128].contiguous(),
        )
        dE.add_(tk_mxfp4_gemm(g_row_q, w_col_chunk, output_scale=scale))
        dC[start:end].copy_(tk_mxfp4_gemm(g_col_q, x_col_q, output_scale=scale))
        del g_row_q, g_col_q, w_col_chunk, w_col_fp4_chunk, q_w_chunk

    if getattr(ctx, "reduce_dE", False):
        torch.distributed.all_reduce(dE, group=ctx.tp_group)
    if getattr(ctx, "prequant_x", False):
        return dE, None, None, None, None, dC, None, None, None, None
    return dE, dC, None, None, None, None


def _mxfp4_p_cache_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    ignore_index: int,
    vocab_size: int,
    mode: int = 1,
):
    mode = int(mode)
    p_target_split = _use_mxfp4_p_target_split()
    p_topk_split = _mxfp4_p_topk_split()
    if p_topk_split and not p_target_split:
        raise RuntimeError("MXFP4 P top-k split also requires target split")

    if p_target_split:
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
        else:
            loss, probs, target_probs = direct_loss_and_probs_target_split(
                logits, targets, valid, int(vocab_size)
            )
            topk_probs = None
            topk_indices = None
        p_row_q, p_col_q = quantize_mxfp4_row_and_col_tk(probs, mode=mode)
        return (
            loss,
            p_row_q.fp4,
            p_row_q.sc,
            p_col_q.fp4,
            p_col_q.sc,
            target_probs,
            topk_indices,
            topk_probs,
        )

    use_direct_p_cache = mode == 1 and use_direct_mxfp4_p_cache()
    if _mxfp4_v4_p_cache_mode() == "auto" and logits.numel() >= _mxfp4_auto_inductor_elements():
        use_direct_p_cache = False
    if use_direct_p_cache:
        try:
            return (
                *mxfp4_direct_p_cache(logits, targets, valid, int(vocab_size)),
                None,
                None,
                None,
            )
        except Exception:
            if os.environ.get("FP4_CCE_V4_STRICT_DIRECT_MXFP4_P_CACHE", "0") == "1":
                raise

    loss, probs = _mxfp4_loss_and_probs(logits, targets, valid, int(ignore_index), int(vocab_size))
    p_row_q, p_col_q = quantize_mxfp4_row_and_col_tk(probs, mode=mode)
    return (
        loss,
        p_row_q.fp4,
        p_row_q.sc,
        p_col_q.fp4,
        p_col_q.sc,
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
class MXFP4CCE_TK_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e_q_fp4, e_q_sc,
                c_q_fp4, c_q_sc,
                c_bf16, e_bf16,
                targets, ignore_index):
        tk = _get_tk_mxfp4()

        M = e_q_fp4.shape[0]
        N = c_q_fp4.shape[0]

        # TK MXFP4 GEMM: logits(M, N) = E @ C^T
        logits = torch.zeros(M, N, dtype=torch.bfloat16, device=e_q_fp4.device)
        tk.mxfp4_gemm(e_q_fp4, e_q_sc, c_q_fp4, c_q_sc, logits)

        # Compute LSE for chunked backward
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
        scale = grad_output / n_valid  # scalar

        # Allocate outputs
        dE = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
        dC = torch.zeros(V, K, dtype=torch.bfloat16, device=device)

        # Chunked backward: loop over vocab in blocks
        CHUNK = 256
        for v_start in range(0, V, CHUNK):
            v_end = min(v_start + CHUNK, V)
            c_chunk = c_bf16[v_start:v_end]        # (chunk, K)

            # Recompute logits for this chunk: (M, chunk)
            logits_chunk = e_bf16 @ c_chunk.T       # BF16 matmul

            # Softmax gradient: P_chunk = exp(logits_chunk - lse)
            grad_chunk = logits_chunk.float()
            grad_chunk.sub_(lse.unsqueeze(1))
            grad_chunk.exp_()

            # Subtract 1 at target positions within this chunk
            target_in_chunk = targets - v_start
            in_range = (target_in_chunk >= 0) & (target_in_chunk < (v_end - v_start)) & valid
            if in_range.any():
                rows_hit = torch.where(in_range)[0]
                cols_hit = target_in_chunk[in_range]
                grad_chunk[rows_hit, cols_hit] -= 1.0

            # Zero out invalid rows
            grad_chunk[~valid] = 0.0

            # Scale by grad_output / n_valid
            grad_chunk = (grad_chunk * scale).to(torch.bfloat16)

            # Accumulate dE and dC
            dE += grad_chunk @ c_chunk              # (M, K)
            dC[v_start:v_end] = grad_chunk.T @ e_bf16  # (chunk, K)

        return None, None, None, None, dC, dE, None, None


class MXFP4CCE_PCache_Function(torch.autograd.Function):
    """v4 CCE prototype with forward softmax P cached in row/col MXFP4."""

    @staticmethod
    def forward(ctx, x, weight, targets, ignore_index, vocab_size, mode):
        mode = int(mode)
        q_x, q_x_col = quantize_mxfp4_row_and_col_tk(x, mode=mode)
        q_w, q_w_col = quantize_mxfp4_row_and_col_tk(weight, mode=mode)

        use_g_cache = _use_mxfp4_g_cache()
        g_target_split = use_g_cache and _use_mxfp4_g_target_split()
        p_target_split = False
        p_top1_split = False
        p_top4_split = False
        use_chunked_g_cache = (
            use_g_cache
            and _use_mxfp4_chunked_logits_g_cache(
                int(q_x.fp4.shape[0]), int(q_w.fp4.shape[0])
            )
        )
        if g_target_split and use_chunked_g_cache:
            raise RuntimeError(
                "MXFP4 G target split requires the direct full-logits producer; "
                "disable FP4_CCE_V4_MXFP4_CHUNKED_LOGITS_G_CACHE"
            )
        if (
            use_chunked_g_cache
        ):
            if _use_mxfp4_chunked_recompute():
                loss, global_lse = _mxfp4_vocab_parallel_chunked_loss_lse(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(vocab_size),
                    0,
                    None,
                )
                ctx.save_for_backward(
                    q_x.fp4, q_x.sc,
                    q_x_col.fp4, q_x_col.sc,
                    q_w.fp4, q_w.sc,
                    q_w_col.fp4, q_w_col.sc,
                    targets,
                    global_lse,
                )
                ctx.ignore_index = int(ignore_index)
                ctx.use_g_cache = True
                ctx.g_target_split = False
                ctx.assume_all_valid = False
                ctx.use_chunked_recompute = True
                ctx.prequant_x = False
                ctx.tp_group = None
                ctx.reduce_dE = False
                ctx.global_vocab_size = int(vocab_size)
                ctx.vocab_start = 0
                ctx.local_valid_cols = max(min(int(vocab_size), int(q_w.fp4.shape[0])), 0)
                ctx.mode = mode
                return loss
            loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = (
                _mxfp4_vocab_parallel_chunked_logits_g_cache(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(vocab_size),
                    0,
                    None,
                    mode=mode,
                )
            )
            ctx.save_for_backward(
                g_row_fp4, g_row_sc,
                g_col_fp4, g_col_sc,
                q_x_col.fp4, q_x_col.sc,
                q_w_col.fp4, q_w_col.sc,
                targets,
            )
            ctx.ignore_index = int(ignore_index)
            ctx.use_g_cache = True
            ctx.g_target_split = False
            ctx.assume_all_valid = False
            return loss

        logits = tk_mxfp4_gemm(q_x, q_w)
        if vocab_size < logits.shape[1]:
            logits = logits.clone()
            logits[:, vocab_size:] = -float("inf")
        assume_all_valid = assume_all_valid_full_vocab(logits, int(vocab_size))
        valid = None

        def get_valid():
            nonlocal valid
            if valid is None:
                valid = targets.ne(ignore_index)
            return valid

        if use_g_cache:
            if g_target_split:
                loss, grad_probs, target_probs = direct_loss_and_probs_target_split(
                    logits, targets, get_valid(), int(vocab_size)
                )
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(
                    grad_probs, mode=mode
                )
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
            elif _use_mxfp4_fused_g_cache() and mode == 1:
                fused_impl = _mxfp4_fused_g_cache_impl()
                if fused_impl == "direct":
                    loss, grad_probs = direct_loss_and_grad_probs(
                        logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                    )
                    g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                    g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                    g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
                elif _use_mxfp4_staged_g_cache() or fused_impl == "staged":
                    loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = mxfp4_staged_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
                else:
                    loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = mxfp4_tiled_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
            elif _use_mxfp4_direct_g_producer() or mode != 1:
                loss, grad_probs = direct_loss_and_grad_probs(
                    logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                )
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
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
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
            if g_target_split:
                ctx.save_for_backward(
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    q_x_col.fp4, q_x_col.sc,
                    q_w_col.fp4, q_w_col.sc,
                    x, weight, targets, target_probs,
                )
            else:
                ctx.save_for_backward(
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    q_x_col.fp4, q_x_col.sc,
                    q_w_col.fp4, q_w_col.sc,
                    targets,
                )
        else:
            (
                loss,
                p_row_fp4,
                p_row_sc,
                p_col_fp4,
                p_col_sc,
                target_probs,
                topk_indices,
                topk_probs,
            ) = _mxfp4_p_cache_from_logits(
                logits,
                targets,
                get_valid(),
                int(ignore_index),
                int(vocab_size),
                mode=mode,
            )
            p_target_split = target_probs is not None
            p_top1_split = topk_indices is not None and topk_indices.dim() == 1
            p_top4_split = topk_indices is not None and topk_indices.dim() == 2
            saved = [
                p_row_fp4, p_row_sc,
                p_col_fp4, p_col_sc,
                q_x_col.fp4, q_x_col.sc,
                q_w_col.fp4, q_w_col.sc,
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
        ctx.p_target_split = bool(p_target_split)
        ctx.p_top1_split = bool(p_top1_split)
        ctx.p_top4_split = bool(p_top4_split)
        ctx.assume_all_valid = bool(assume_all_valid)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, "use_chunked_recompute", False):
            return _mxfp4_chunked_recompute_backward(ctx, grad_output)

        if ctx.use_g_cache:
            if ctx.g_target_split:
                (
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    x_col_fp4, x_col_sc,
                    w_col_fp4, w_col_sc,
                    x, weight, targets, target_probs,
                ) = ctx.saved_tensors
            else:
                (
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    x_col_fp4, x_col_sc,
                    w_col_fp4, w_col_sc,
                    targets,
                ) = ctx.saved_tensors
        else:
            if ctx.p_target_split:
                if ctx.p_top1_split or ctx.p_top4_split:
                    (
                        p_row_fp4, p_row_sc,
                        p_col_fp4, p_col_sc,
                        x_col_fp4, x_col_sc,
                        w_col_fp4, w_col_sc,
                        x, weight, targets, target_probs,
                        topk_indices, topk_probs,
                    ) = ctx.saved_tensors
                else:
                    (
                        p_row_fp4, p_row_sc,
                        p_col_fp4, p_col_sc,
                        x_col_fp4, x_col_sc,
                        w_col_fp4, w_col_sc,
                        x, weight, targets, target_probs,
                    ) = ctx.saved_tensors
            else:
                (
                    p_row_fp4, p_row_sc,
                    p_col_fp4, p_col_sc,
                    x_col_fp4, x_col_sc,
                    w_col_fp4, w_col_sc,
                    x, weight, targets,
                ) = ctx.saved_tensors

        if getattr(ctx, "assume_all_valid", False):
            scale = (grad_output.float() / float(targets.numel())).reshape(())
        else:
            n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
            scale = (grad_output.float() / n_valid).reshape(())

        if ctx.use_g_cache:
            g_row = MXFP4Quantized(g_row_fp4, g_row_sc)
            g_col = MXFP4Quantized(g_col_fp4, g_col_sc)
            x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
            w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)
            dE = tk_mxfp4_gemm(g_row, w_col_q, output_scale=scale)
            dC = tk_mxfp4_gemm(g_col, x_col_q, output_scale=scale)
            if ctx.g_target_split:
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

        p_row = MXFP4Quantized(p_row_fp4, p_row_sc)
        p_col = MXFP4Quantized(p_col_fp4, p_col_sc)
        x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
        w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)

        if ctx.p_target_split:
            dE = tk_mxfp4_gemm(p_row, w_col_q, output_scale=scale)
            dC = tk_mxfp4_gemm(p_col, x_col_q, output_scale=scale)
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
            else:
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
            dE = tk_mxfp4_gemm(p_row, w_col_q)
            dC = tk_mxfp4_gemm(p_col, x_col_q, output_scale=scale)
            sparse_correct_scaled_dE(
                dE, dC, x, weight, targets, scale, ctx.ignore_index
            )

        return dE, dC, None, None, None, None


class MXFP4CCE_PCache_PrequantX_Function(torch.autograd.Function):
    """v4 CCE with x/x.T MXFP4 quantization supplied by an upstream producer."""

    @staticmethod
    def forward(
        ctx,
        x,
        x_fp4,
        x_sc,
        x_col_fp4,
        x_col_sc,
        weight,
        targets,
        ignore_index,
        vocab_size,
        mode,
    ):
        mode = int(mode)
        q_x = MXFP4Quantized(x_fp4, x_sc)
        q_x_col = MXFP4Quantized(x_col_fp4, x_col_sc)
        q_w, q_w_col = quantize_mxfp4_row_and_col_tk(weight, mode=mode)

        use_g_cache = _use_mxfp4_g_cache()
        g_target_split = use_g_cache and _use_mxfp4_g_target_split()
        p_target_split = False
        p_top1_split = False
        p_top4_split = False
        use_chunked_g_cache = (
            use_g_cache
            and _use_mxfp4_chunked_logits_g_cache(
                int(q_x.fp4.shape[0]), int(q_w.fp4.shape[0])
            )
        )
        if g_target_split and use_chunked_g_cache:
            raise RuntimeError(
                "MXFP4 G target split requires the direct full-logits producer; "
                "disable FP4_CCE_V4_MXFP4_CHUNKED_LOGITS_G_CACHE"
            )
        if (
            use_chunked_g_cache
        ):
            if _use_mxfp4_chunked_recompute():
                loss, global_lse = _mxfp4_vocab_parallel_chunked_loss_lse(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(vocab_size),
                    0,
                    None,
                )
                ctx.save_for_backward(
                    q_x.fp4, q_x.sc,
                    x_col_fp4, x_col_sc,
                    q_w.fp4, q_w.sc,
                    q_w_col.fp4, q_w_col.sc,
                    targets,
                    global_lse,
                )
                ctx.ignore_index = int(ignore_index)
                ctx.use_g_cache = True
                ctx.g_target_split = False
                ctx.assume_all_valid = False
                ctx.use_chunked_recompute = True
                ctx.prequant_x = True
                ctx.tp_group = None
                ctx.reduce_dE = False
                ctx.global_vocab_size = int(vocab_size)
                ctx.vocab_start = 0
                ctx.local_valid_cols = max(min(int(vocab_size), int(q_w.fp4.shape[0])), 0)
                ctx.mode = mode
                return loss
            loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = (
                _mxfp4_vocab_parallel_chunked_logits_g_cache(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(vocab_size),
                    0,
                    None,
                    mode=mode,
                )
            )
            ctx.save_for_backward(
                g_row_fp4, g_row_sc,
                g_col_fp4, g_col_sc,
                x_col_fp4, x_col_sc,
                q_w_col.fp4, q_w_col.sc,
                targets,
            )
            ctx.ignore_index = int(ignore_index)
            ctx.use_g_cache = True
            ctx.g_target_split = False
            ctx.assume_all_valid = False
            return loss

        logits = tk_mxfp4_gemm(q_x, q_w)
        if vocab_size < logits.shape[1]:
            logits = logits.clone()
            logits[:, vocab_size:] = -float("inf")
        assume_all_valid = assume_all_valid_full_vocab(logits, int(vocab_size))
        valid = None

        def get_valid():
            nonlocal valid
            if valid is None:
                valid = targets.ne(ignore_index)
            return valid

        if use_g_cache:
            if g_target_split:
                loss, grad_probs, target_probs = direct_loss_and_probs_target_split(
                    logits, targets, get_valid(), int(vocab_size)
                )
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(
                    grad_probs, mode=mode
                )
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
            elif _use_mxfp4_fused_g_cache() and mode == 1:
                fused_impl = _mxfp4_fused_g_cache_impl()
                if fused_impl == "direct":
                    loss, grad_probs = direct_loss_and_grad_probs(
                        logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                    )
                    g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                    g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                    g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
                elif _use_mxfp4_staged_g_cache() or fused_impl == "staged":
                    loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = mxfp4_staged_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
                else:
                    loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = mxfp4_tiled_g_cache(
                        logits, targets, get_valid(), int(vocab_size)
                    )
            elif _use_mxfp4_direct_g_producer() or mode != 1:
                loss, grad_probs = direct_loss_and_grad_probs(
                    logits, targets, targets if assume_all_valid else get_valid(), int(vocab_size)
                )
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
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
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
            if g_target_split:
                ctx.save_for_backward(
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    x_col_fp4, x_col_sc,
                    q_w_col.fp4, q_w_col.sc,
                    x, weight, targets, target_probs,
                )
            else:
                ctx.save_for_backward(
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    x_col_fp4, x_col_sc,
                    q_w_col.fp4, q_w_col.sc,
                    targets,
                )
        else:
            (
                loss,
                p_row_fp4,
                p_row_sc,
                p_col_fp4,
                p_col_sc,
                target_probs,
                topk_indices,
                topk_probs,
            ) = _mxfp4_p_cache_from_logits(
                logits,
                targets,
                get_valid(),
                int(ignore_index),
                int(vocab_size),
                mode=mode,
            )
            p_target_split = target_probs is not None
            p_top1_split = topk_indices is not None and topk_indices.dim() == 1
            p_top4_split = topk_indices is not None and topk_indices.dim() == 2
            saved = [
                p_row_fp4, p_row_sc,
                p_col_fp4, p_col_sc,
                x_col_fp4, x_col_sc,
                q_w_col.fp4, q_w_col.sc,
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
        ctx.p_target_split = bool(p_target_split)
        ctx.p_top1_split = bool(p_top1_split)
        ctx.p_top4_split = bool(p_top4_split)
        ctx.assume_all_valid = bool(assume_all_valid)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, "use_chunked_recompute", False):
            return _mxfp4_chunked_recompute_backward(ctx, grad_output)

        if ctx.use_g_cache:
            if ctx.g_target_split:
                (
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    x_col_fp4, x_col_sc,
                    w_col_fp4, w_col_sc,
                    x, weight, targets, target_probs,
                ) = ctx.saved_tensors
            else:
                (
                    g_row_fp4, g_row_sc,
                    g_col_fp4, g_col_sc,
                    x_col_fp4, x_col_sc,
                    w_col_fp4, w_col_sc,
                    targets,
                ) = ctx.saved_tensors
        else:
            if ctx.p_target_split:
                if ctx.p_top1_split or ctx.p_top4_split:
                    (
                        p_row_fp4, p_row_sc,
                        p_col_fp4, p_col_sc,
                        x_col_fp4, x_col_sc,
                        w_col_fp4, w_col_sc,
                        x, weight, targets, target_probs,
                        topk_indices, topk_probs,
                    ) = ctx.saved_tensors
                else:
                    (
                        p_row_fp4, p_row_sc,
                        p_col_fp4, p_col_sc,
                        x_col_fp4, x_col_sc,
                        w_col_fp4, w_col_sc,
                        x, weight, targets, target_probs,
                    ) = ctx.saved_tensors
            else:
                (
                    p_row_fp4, p_row_sc,
                    p_col_fp4, p_col_sc,
                    x_col_fp4, x_col_sc,
                    w_col_fp4, w_col_sc,
                    x, weight, targets,
                ) = ctx.saved_tensors

        if getattr(ctx, "assume_all_valid", False):
            scale = (grad_output.float() / float(targets.numel())).reshape(())
        else:
            n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
            scale = (grad_output.float() / n_valid).reshape(())

        if ctx.use_g_cache:
            g_row = MXFP4Quantized(g_row_fp4, g_row_sc)
            g_col = MXFP4Quantized(g_col_fp4, g_col_sc)
            x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
            w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)
            dE = tk_mxfp4_gemm(g_row, w_col_q, output_scale=scale)
            dC = tk_mxfp4_gemm(g_col, x_col_q, output_scale=scale)
            if ctx.g_target_split:
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
            return dE, None, None, None, None, dC, None, None, None, None

        p_row = MXFP4Quantized(p_row_fp4, p_row_sc)
        p_col = MXFP4Quantized(p_col_fp4, p_col_sc)
        x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
        w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)

        if ctx.p_target_split:
            dE = tk_mxfp4_gemm(p_row, w_col_q, output_scale=scale)
            dC = tk_mxfp4_gemm(p_col, x_col_q, output_scale=scale)
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
            else:
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
            dE = tk_mxfp4_gemm(p_row, w_col_q)
            dC = tk_mxfp4_gemm(p_col, x_col_q, output_scale=scale)
            sparse_correct_scaled_dE(
                dE, dC, x, weight, targets, scale, ctx.ignore_index
            )

        return dE, None, None, None, None, dC, None, None, None, None


class MXFP4CCE_NativePrecision_Function(torch.autograd.Function):
    """Native CUDA/TK CCE with independently selected forward/backward GEMMs."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        targets,
        ignore_index,
        vocab_size,
        mode,
        forward_precision,
        backward_precision,
    ):
        mode = int(mode)
        forward_precision = str(forward_precision)
        backward_precision = str(backward_precision)
        if mode != 1:
            raise ValueError("native precision ablation requires MXFP4 encode mode")
        if forward_precision not in {"bf16", "fp4"}:
            raise ValueError(f"unsupported forward precision: {forward_precision}")
        if backward_precision not in {"bf16", "fp4"}:
            raise ValueError(f"unsupported backward precision: {backward_precision}")

        valid, valid_count = valid_mask_count_cuda(targets, int(ignore_index))
        fp4_forward = forward_precision == "fp4"
        fp4_backward = backward_precision == "fp4"

        q_x = q_w = q_x_col = q_w_col = None
        if fp4_backward:
            q_x, q_x_col = quantize_mxfp4_row_and_col_tk(x, mode=mode)
            q_w, q_w_col = quantize_mxfp4_row_and_col_tk(weight, mode=mode)
        elif fp4_forward:
            q_x = quantize_mxfp4_tk(x, keep_bf16=False, mode=mode)
            q_w = quantize_mxfp4_tk(weight, keep_bf16=False, mode=mode)

        logits = (
            tk_mxfp4_gemm(q_x, q_w)
            if fp4_forward
            else bf16_logits_cuda(x, weight)
        )

        if fp4_backward:
            loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = (
                mxfp4_staged_g_cache(
                    logits,
                    targets,
                    valid,
                    int(vocab_size),
                )
            )
            ctx.save_for_backward(
                g_row_fp4,
                g_row_sc,
                g_col_fp4,
                g_col_sc,
                q_x_col.fp4,
                q_x_col.sc,
                q_w_col.fp4,
                q_w_col.sc,
                valid_count,
            )
        else:
            loss, grad_logits = direct_loss_and_grad_probs(
                logits,
                targets,
                valid,
                int(vocab_size),
            )
            ctx.save_for_backward(
                grad_logits,
                x,
                weight,
                valid_count,
            )

        ctx.fp4_backward = fp4_backward
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.fp4_backward:
            (
                g_row_fp4,
                g_row_sc,
                g_col_fp4,
                g_col_sc,
                x_col_fp4,
                x_col_sc,
                w_col_fp4,
                w_col_sc,
                valid_count,
            ) = ctx.saved_tensors
            scale = backward_scale_cuda(grad_output, valid_count)
            g_row = MXFP4Quantized(g_row_fp4, g_row_sc)
            g_col = MXFP4Quantized(g_col_fp4, g_col_sc)
            x_col = MXFP4Quantized(x_col_fp4, x_col_sc)
            w_col = MXFP4Quantized(w_col_fp4, w_col_sc)
            d_x = tk_mxfp4_gemm(g_row, w_col, output_scale=scale)
            d_weight = tk_mxfp4_gemm(g_col, x_col, output_scale=scale)
        else:
            grad_logits, x, weight, valid_count = ctx.saved_tensors
            d_x, d_weight = bf16_tail_grads_cuda(
                grad_logits,
                x,
                weight,
                grad_output,
                valid_count,
            )

        return d_x, d_weight, None, None, None, None, None, None


class MXFP4CCE_VocabParallel_Function(torch.autograd.Function):
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
        mode,
    ):
        mode = int(mode)
        q_x, q_x_col = quantize_mxfp4_row_and_col_tk(x, mode=mode)
        q_w, q_w_col = quantize_mxfp4_row_and_col_tk(weight, mode=mode)
        valid = targets.ne(int(ignore_index))
        if (
            use_mxfp4_vocab_parallel_direct_g_cache()
            and _use_mxfp4_chunked_logits_g_cache(int(q_x.fp4.shape[0]), int(q_w.fp4.shape[0]))
        ):
            loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = (
                _mxfp4_vocab_parallel_chunked_logits_g_cache(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(global_vocab_size),
                    int(vocab_start),
                    tp_group,
                    mode=mode,
                )
            )
        else:
            logits = tk_mxfp4_gemm(q_x, q_w)
            if use_mxfp4_vocab_parallel_direct_g_cache():
                loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = mxfp4_vocab_parallel_tiled_g_cache(
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
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
        ctx.save_for_backward(
            g_row_fp4,
            g_row_sc,
            g_col_fp4,
            g_col_sc,
            q_x_col.fp4,
            q_x_col.sc,
            q_w_col.fp4,
            q_w_col.sc,
            targets,
        )
        ctx.ignore_index = int(ignore_index)
        ctx.tp_group = tp_group
        ctx.reduce_dE = bool(reduce_dE)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            g_row_fp4,
            g_row_sc,
            g_col_fp4,
            g_col_sc,
            x_col_fp4,
            x_col_sc,
            w_col_fp4,
            w_col_sc,
            targets,
        ) = ctx.saved_tensors

        n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
        scale = (grad_output.float() / n_valid).reshape(())
        g_row = MXFP4Quantized(g_row_fp4, g_row_sc)
        g_col = MXFP4Quantized(g_col_fp4, g_col_sc)
        x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
        w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)

        # Return this rank's vocab-shard contribution. When Bridge feeds a
        # replicated hidden state, overlap the required TP input-gradient
        # reduction with the independent weight-gradient GEMM.
        dE = tk_mxfp4_gemm(g_row, w_col_q, output_scale=scale)
        dE_reduce = None
        if ctx.reduce_dE:
            dE_reduce = torch.distributed.all_reduce(dE, group=ctx.tp_group, async_op=True)
        dC = tk_mxfp4_gemm(g_col, x_col_q, output_scale=scale)
        if dE_reduce is not None:
            dE_reduce.wait()
        return dE, dC, None, None, None, None, None, None, None


class MXFP4CCE_VocabParallel_PrequantX_Function(torch.autograd.Function):
    """Bridge TP CCE with MXFP4 x/x.T supplied by an upstream producer."""

    @staticmethod
    def forward(
        ctx,
        x,
        x_fp4,
        x_sc,
        x_col_fp4,
        x_col_sc,
        weight,
        targets,
        ignore_index,
        global_vocab_size,
        vocab_start,
        tp_group,
        reduce_dE,
        mode,
    ):
        mode = int(mode)
        q_x = MXFP4Quantized(x_fp4, x_sc)
        q_x_col = MXFP4Quantized(x_col_fp4, x_col_sc)
        q_w, q_w_col = quantize_mxfp4_row_and_col_tk(weight, mode=mode)
        valid = targets.ne(int(ignore_index))
        if (
            use_mxfp4_vocab_parallel_direct_g_cache()
            and _use_mxfp4_chunked_logits_g_cache(int(q_x.fp4.shape[0]), int(q_w.fp4.shape[0]))
        ):
            loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = (
                _mxfp4_vocab_parallel_chunked_logits_g_cache(
                    q_x,
                    q_w,
                    targets,
                    int(ignore_index),
                    int(global_vocab_size),
                    int(vocab_start),
                    tp_group,
                    mode=mode,
                )
            )
        else:
            logits = tk_mxfp4_gemm(q_x, q_w)
            if use_mxfp4_vocab_parallel_direct_g_cache():
                loss, g_row_fp4, g_row_sc, g_col_fp4, g_col_sc = mxfp4_vocab_parallel_tiled_g_cache(
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
                g_row_q, g_col_q = quantize_mxfp4_row_and_col_tk(grad_probs, mode=mode)
                g_row_fp4, g_row_sc = g_row_q.fp4, g_row_q.sc
                g_col_fp4, g_col_sc = g_col_q.fp4, g_col_q.sc
        ctx.save_for_backward(
            g_row_fp4,
            g_row_sc,
            g_col_fp4,
            g_col_sc,
            q_x_col.fp4,
            q_x_col.sc,
            q_w_col.fp4,
            q_w_col.sc,
            targets,
        )
        ctx.ignore_index = int(ignore_index)
        ctx.tp_group = tp_group
        ctx.reduce_dE = bool(reduce_dE)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (
            g_row_fp4,
            g_row_sc,
            g_col_fp4,
            g_col_sc,
            x_col_fp4,
            x_col_sc,
            w_col_fp4,
            w_col_sc,
            targets,
        ) = ctx.saved_tensors

        n_valid = targets.ne(ctx.ignore_index).sum().clamp(min=1).float()
        scale = (grad_output.float() / n_valid).reshape(())
        g_row = MXFP4Quantized(g_row_fp4, g_row_sc)
        g_col = MXFP4Quantized(g_col_fp4, g_col_sc)
        x_col_q = MXFP4Quantized(x_col_fp4, x_col_sc)
        w_col_q = MXFP4Quantized(w_col_fp4, w_col_sc)

        dE = tk_mxfp4_gemm(g_row, w_col_q, output_scale=scale)
        dE_reduce = None
        if ctx.reduce_dE:
            dE_reduce = torch.distributed.all_reduce(dE, group=ctx.tp_group, async_op=True)
        dC = tk_mxfp4_gemm(g_col, x_col_q, output_scale=scale)
        if dE_reduce is not None:
            dE_reduce.wait()
        return dE, None, None, None, None, dC, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def mxfp4_cce_tk(
    x: torch.Tensor,       # (N, D) BF16 embeddings
    weight: torch.Tensor,   # (V, D) BF16 classifier weights
    targets: torch.Tensor,  # (N,) int64
    ignore_index: int = -100,
) -> torch.Tensor:
    """MXFP4 Cross-Entropy using ThunderKittens GEMM.

    Quantizes both input and weight to MXFP4, computes logits via TK GEMM,
    then applies F.cross_entropy.
    """
    M, K = x.shape
    V = weight.shape[0]

    # Pad M to multiple of 128, V to multiple of 256 (TK tile Nb)
    M_ALIGN = 128
    V_ALIGN = 256

    M_pad = ((M + M_ALIGN - 1) // M_ALIGN) * M_ALIGN
    V_pad = ((V + V_ALIGN - 1) // V_ALIGN) * V_ALIGN

    if M_pad != M:
        x = F.pad(x, (0, 0, 0, M_pad - M))
        targets = F.pad(targets, (0, M_pad - M), value=ignore_index)
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    q_x = quantize_mxfp4_tk(x)
    q_w = quantize_mxfp4_tk(weight)

    return MXFP4CCE_TK_Function.apply(
        q_x.fp4, q_x.sc,
        q_w.fp4, q_w.sc,
        weight, x,
        targets.to(torch.int64), ignore_index,
    )


def mxfp4_cce_tk_v4_pcache(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    mode: int = 1,
) -> torch.Tensor:
    """MXFP4 CCE v4 prototype using a quantized softmax-probability cache."""
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

    return MXFP4CCE_PCache_Function.apply(
        x, weight, targets.to(torch.int64), int(ignore_index), int(V), int(mode)
    )


def mxfp4_cce_tk_native_precision(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    mode: int = 1,
    forward_precision: str = "bf16",
    backward_precision: str = "bf16",
) -> torch.Tensor:
    """Run a production-shape CCE precision cell using only CUDA/TK kernels."""
    M, K = x.shape
    V = weight.shape[0]
    if M % 128 or K % 128 or V % 256:
        raise ValueError(
            "native precision CCE requires M/K multiples of 128 and V a "
            f"multiple of 256; got M={M}, K={K}, V={V}"
        )
    if targets.dtype != torch.int64:
        raise TypeError("native precision CCE targets must be int64")
    return MXFP4CCE_NativePrecision_Function.apply(
        x,
        weight,
        targets,
        int(ignore_index),
        int(V),
        int(mode),
        str(forward_precision),
        str(backward_precision),
    )


def mxfp4_cce_tk_v4_pcache_prequantized_x(
    x: torch.Tensor,
    x_q: MXFP4Quantized,
    x_col_q: MXFP4Quantized,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    mode: int = 1,
) -> torch.Tensor:
    """MXFP4 CCE v4 with row/col quantized x supplied by the producer.

    `x` is still required for exact sparse label correction and for the
    gradient returned to the producer.
    """
    M, _K = x.shape
    V = weight.shape[0]
    M_ALIGN = 128
    V_ALIGN = 256

    M_pad = ((M + M_ALIGN - 1) // M_ALIGN) * M_ALIGN
    V_pad = ((V + V_ALIGN - 1) // V_ALIGN) * V_ALIGN
    if M_pad != M:
        raise ValueError("prequantized x path requires caller-padded M to a multiple of 128")
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    if x_q.fp4.shape[0] != x.shape[0] or x_col_q.fp4.shape[0] != x.shape[1]:
        raise ValueError("prequantized x tensors do not match x shape")

    return MXFP4CCE_PCache_PrequantX_Function.apply(
        x,
        x_q.fp4,
        x_q.sc,
        x_col_q.fp4,
        x_col_q.sc,
        weight,
        targets.to(torch.int64),
        int(ignore_index),
        int(V),
        int(mode),
    )


def mxfp4_cce_tk_v4_vocab_parallel(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    global_vocab_size: int | None = None,
    vocab_start: int = 0,
    tp_group=None,
    reduce_dE: bool = False,
    mode: int = 1,
) -> torch.Tensor:
    """MXFP4 v4 CCE for Bridge vocab-parallel output weights."""
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

    return MXFP4CCE_VocabParallel_Function.apply(
        x,
        weight,
        targets.to(torch.int64),
        int(ignore_index),
        int(global_vocab_size),
        int(vocab_start),
        tp_group,
        bool(reduce_dE),
        int(mode),
    )


def mxfp4_cce_tk_v4_vocab_parallel_prequantized_x(
    x: torch.Tensor,
    x_q: MXFP4Quantized,
    x_col_q: MXFP4Quantized,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    global_vocab_size: int | None = None,
    vocab_start: int = 0,
    tp_group=None,
    reduce_dE: bool = False,
    mode: int = 1,
) -> torch.Tensor:
    """MXFP4 v4 vocab-parallel CCE with row/col quantized x supplied upstream."""
    M, _K = x.shape
    V = weight.shape[0]
    M_ALIGN = 128
    V_ALIGN = 256

    M_pad = ((M + M_ALIGN - 1) // M_ALIGN) * M_ALIGN
    V_pad = ((V + V_ALIGN - 1) // V_ALIGN) * V_ALIGN
    if M_pad != M:
        raise ValueError("prequantized vocab-parallel x path requires M to be a multiple of 128")
    if V_pad != V:
        weight = F.pad(weight, (0, 0, 0, V_pad - V))

    if x_q.fp4.shape[0] != x.shape[0] or x_col_q.fp4.shape[0] != x.shape[1]:
        raise ValueError("prequantized x tensors do not match x shape")

    if global_vocab_size is None:
        global_vocab_size = int(vocab_start) + int(V)

    return MXFP4CCE_VocabParallel_PrequantX_Function.apply(
        x,
        x_q.fp4,
        x_q.sc,
        x_col_q.fp4,
        x_col_q.sc,
        weight,
        targets.to(torch.int64),
        int(ignore_index),
        int(global_vocab_size),
        int(vocab_start),
        tp_group,
        bool(reduce_dE),
        int(mode),
    )
