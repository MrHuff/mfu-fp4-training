"""
NVFP4 Chunked Cross-Entropy (CCE) — On-the-fly NVFP4 GEMM + Cross-Entropy

Instead of materializing full [M, V] logits, this computes cross-entropy
in chunks along the vocab dimension using online log-sum-exp reduction.

Memory: O(M × chunk_size) instead of O(M × V)
At DeepSeek scale (4K×256K): 32MB vs 2GB.

Algorithm:
  1. Quantize e and c to NVFP4 (once, up front)  
  2. For each vocab chunk [v_start, v_end):
     - Slice pre-quantized c_fp4[chunk], c_sc[chunk] (zero-copy)
     - GEMM: logits_chunk = e_fp4 @ c_fp4_chunk^T  → [M, chunk_size]
     - Extract neg_correct_logit for targets in this chunk
     - Compute chunk_lse = logsumexp(logits_chunk, dim=1)
     - Merge with running_lse via logaddexp
  3. loss = (neg_correct_logit + running_lse).mean()
"""

import os
import sys
import torch
import torch.nn.functional as F

# Ensure paths are set up
_base = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_base)
if _base not in sys.path:
    sys.path.insert(0, _base)

from nvfp4_cce_tk import _get_tk_nvfp4, quantize_nvfp4_tk, NVFP4Quantized


# ---------------------------------------------------------------------------
# Core: Chunked NVFP4 GEMM + online cross-entropy
# ---------------------------------------------------------------------------

def nvfp4_chunked_cce(
    e: torch.Tensor,        # [M, K] BF16 embeddings
    c: torch.Tensor,        # [V, K] BF16 classifier weights
    targets: torch.Tensor,  # [M] int64 target indices
    chunk_size: int = 4096, # vocab chunk size (must be multiple of 256)
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Chunked NVFP4 Cross-Entropy: never materializes [M, V] logits.
    
    Returns scalar loss (mean reduction).
    """
    assert e.ndim == 2 and c.ndim == 2
    assert e.size(1) == c.size(1), f"K mismatch: e={e.size(1)} c={c.size(1)}"
    assert chunk_size % 256 == 0, f"chunk_size must be multiple of 256, got {chunk_size}"
    
    M, K = e.shape
    V = c.size(0)
    
    # Pad M and K to multiples of 256/128 for TK GEMM
    M_pad = ((M + 255) // 256) * 256
    K_pad = ((K + 255) // 256) * 256
    V_pad = ((V + chunk_size - 1) // chunk_size) * chunk_size  # pad V to multiple of chunk_size
    V_pad = ((V_pad + 255) // 256) * 256  # also to 256
    
    # Pad embeddings
    if M != M_pad or K != K_pad:
        e_padded = torch.zeros(M_pad, K_pad, dtype=e.dtype, device=e.device)
        e_padded[:M, :K] = e
    else:
        e_padded = e
    
    # Pad weights to V_pad
    if K != K_pad or V != V_pad:
        c_padded = torch.zeros(V_pad, K_pad, dtype=c.dtype, device=c.device)
        c_padded[:V, :K] = c
    else:
        c_padded = c
    
    tk = _get_tk_nvfp4()
    
    # Quantize ONCE: e and ALL of c
    e_q = quantize_nvfp4_tk(e_padded)
    c_q = quantize_nvfp4_tk(c_padded)
    
    # c_q.fp4: [V_pad, K_pad//2]
    # c_q.sc:  [V_pad//128, K_pad//64, 512] (or similar layout)
    # c_q.sg:  [1] float32
    
    # Running online log-sum-exp state
    running_lse = torch.full((M,), -float('inf'), dtype=torch.float32, device=e.device)
    neg_correct_logit = torch.zeros(M, dtype=torch.float32, device=e.device)
    
    # Valid mask (for ignore_index)
    valid = targets.ne(ignore_index)
    
    # Pre-allocate output buffer (reused across chunks)
    logits_buf = torch.empty(M_pad, chunk_size, dtype=torch.bfloat16, device=e.device)
    
    num_chunks = V_pad // chunk_size
    
    for chunk_idx in range(num_chunks):
        v_start = chunk_idx * chunk_size
        v_end = v_start + chunk_size
        actual_end = min(v_end, V)  # actual vocab boundary
        
        # Slice pre-quantized weight data (zero-copy views)
        row_start = v_start
        row_end = v_end
        fp4_chunk = c_q.fp4[row_start:row_end].contiguous()
        
        # Scale layout: [V_pad//128, ...] → slice first dim
        sc_row_start = row_start // 128
        sc_row_end = row_end // 128
        sc_chunk = c_q.sc[sc_row_start:sc_row_end].contiguous()
        
        # GEMM: [M_pad, K] @ [chunk_size, K]^T → [M_pad, chunk_size]
        logits_buf.zero_()
        tk.nvfp4_gemm(e_q.fp4, e_q.sc, e_q.sg,
                       fp4_chunk, sc_chunk, c_q.sg,
                       logits_buf)
        
        # Extract valid region [M, actual_chunk]
        actual_chunk = actual_end - v_start
        logits_chunk = logits_buf[:M, :actual_chunk].float()
        
        # Extract correct-class logits for targets in this chunk
        in_chunk = (targets >= v_start) & (targets < actual_end) & valid
        if in_chunk.any():
            local_targets = targets[in_chunk] - v_start
            rows = torch.where(in_chunk)[0]
            neg_correct_logit[rows] = -logits_chunk[rows, local_targets]
        
        # Online LSE update: logaddexp(running_lse, logsumexp(chunk))
        if actual_chunk < chunk_size:
            # Mask padding to -inf
            logits_chunk_padded = logits_buf[:M, :chunk_size].float()
            logits_chunk_padded[:, actual_chunk:] = -float('inf')
            chunk_lse = torch.logsumexp(logits_chunk_padded, dim=1)
        else:
            chunk_lse = torch.logsumexp(logits_chunk, dim=1)
        
        running_lse = torch.logaddexp(running_lse, chunk_lse)
    
    # NLL per row = neg_correct_logit + lse
    nll = neg_correct_logit + running_lse
    
    # Apply ignore_index masking
    if not valid.all():
        nll = nll[valid]
    
    return nll.mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nvfp4_chunked_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Public API: chunked NVFP4 cross-entropy (forward only)."""
    return nvfp4_chunked_cce(e.detach(), c.detach(), targets, chunk_size, ignore_index)
