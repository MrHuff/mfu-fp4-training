#!/usr/bin/env python3
"""Stage-by-stage numerics comparison for fused FP4 attention wrappers.

This compares the fused attention wrapper against the unfused torchtitan
attention path on identical weights and inputs. It is intended for numerics
debugging after speed recovery, not benchmarking.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.abspath(
    os.environ.get(
        "LOW_BITS_TRAINING_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
)
TOOLS_ROOT = os.path.join(REPO_ROOT, "tools")
TORCHTITAN_ROOT = os.path.join(REPO_ROOT, "torchtitan_submodule")
FALLBACK_TORCHTITAN_ROOT = "/opt/mfu/EXTERNAL_PATH"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TOOLS_ROOT not in sys.path:
    sys.path.insert(0, TOOLS_ROOT)
if TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, TORCHTITAN_ROOT)
if os.path.isdir(FALLBACK_TORCHTITAN_ROOT) and FALLBACK_TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, FALLBACK_TORCHTITAN_ROOT)

from bench_synth_1b_fp4 import _model_args_for_flavor, configure_env, _preinit_cuda, _set_device_with_retry
from low_bits_training.quantization.fp4_converter import _FusedAttentionWrapper
from low_bits_training.quantization import fused_te_linear as ftl
from low_bits_training.quantization.fused_te_linear import FusedAttentionFP4_TE, FusedAttentionFP4_TK
from torchtitan.models.llama3.model.model import (
    TransformerBlock,
    apply_rotary_emb,
    precompute_freqs_cis,
    repeat_kv,
)


VALID_BACKENDS = {"te", "tk", "localcta", "localcta_fused"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare fused attention numerics stage by stage.")
    parser.add_argument("--flavor", choices=["1B", "1B_legacy", "8B"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--reference-backend",
        choices=["bf16", "te"],
        default="bf16",
        help="Use exact BF16 linears or TE FP4 as the reference path.",
    )
    parser.add_argument(
        "--backend-combos",
        nargs="+",
        default=["te/te", "localcta_fused/te", "te/localcta_fused", "localcta/te", "te/localcta"],
        help="attention/ffn backend pairs to label and configure",
    )
    parser.add_argument(
        "--qkv-decompose",
        action="store_true",
        help="On localCTA attention combos, capture the real QKV payloads and compare "
             "reconstructed-BF16 matmul against the actual grouped GEMM output.",
    )
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def _metric(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    diff = a_flat - b_flat
    denom = float(a_flat.norm() * b_flat.norm())
    if denom == 0.0:
        cos = 1.0 if float((a_flat - b_flat).abs().max()) == 0.0 else float("nan")
    else:
        cos = float(F.cosine_similarity(a_flat, b_flat, dim=0).item())
    mse = float((diff * diff).mean().item())
    rmse = mse ** 0.5
    ref_rms = float((a_flat * a_flat).mean().sqrt().item())
    test_rms = float((b_flat * b_flat).mean().sqrt().item())
    ref_mean_abs = float(a_flat.abs().mean().item())
    test_mean_abs = float(b_flat.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    return {
        "cos": cos,
        "mse": mse,
        "rmse": rmse,
        "ref_rms": ref_rms,
        "test_rms": test_rms,
        "ref_mean_abs": ref_mean_abs,
        "test_mean_abs": test_mean_abs,
        "rel_rmse_vs_ref_rms": rmse / ref_rms if ref_rms > 0.0 else float("inf"),
        "test_rms_vs_ref_rms": test_rms / ref_rms if ref_rms > 0.0 else float("inf"),
        "rel_max_vs_ref_mean_abs": max_abs / ref_mean_abs if ref_mean_abs > 0.0 else float("inf"),
        "maxerr": max_abs,
        "finite_ref": bool(torch.isfinite(a).all().item()),
        "finite_test": bool(torch.isfinite(b).all().item()),
    }


def _metric_is_nonfinite(metric: dict[str, Any]) -> bool:
    return not metric.get("finite_ref", True) or not metric.get("finite_test", True)


def _metric_is_catastrophic(metric: dict[str, Any], cos_threshold: float = 0.95) -> bool:
    if _metric_is_nonfinite(metric):
        return True
    cos = metric.get("cos")
    if cos is None:
        return False
    return not torch.isfinite(torch.tensor(float(cos))).item() or float(cos) < cos_threshold


def _summarize_metric_chain(
    metrics: dict[str, dict[str, Any]],
    order: list[str],
    *,
    cos_threshold: float = 0.95,
) -> dict[str, Any]:
    first_nonfinite = None
    first_catastrophic = None
    for name in order:
        metric = metrics.get(name)
        if metric is None:
            continue
        if first_nonfinite is None and _metric_is_nonfinite(metric):
            first_nonfinite = {"name": name, "metric": metric}
        if first_catastrophic is None and _metric_is_catastrophic(metric, cos_threshold=cos_threshold):
            first_catastrophic = {"name": name, "metric": metric}
        if first_nonfinite is not None and first_catastrophic is not None:
            break
    return {
        "order": order,
        "catastrophic_cos_threshold": cos_threshold,
        "first_nonfinite_tensor": None if first_nonfinite is None else first_nonfinite["name"],
        "first_nonfinite_metric": None if first_nonfinite is None else first_nonfinite["metric"],
        "first_catastrophic_tensor": None if first_catastrophic is None else first_catastrophic["name"],
        "first_catastrophic_metric": None if first_catastrophic is None else first_catastrophic["metric"],
    }


def _head_alignment_summary(ref: torch.Tensor, other: torch.Tensor) -> dict[str, Any]:
    """Return simple per-head alignment stats for tensors shaped (B, S, H, D)."""
    if ref.dim() != 4 or other.dim() != 4 or ref.shape[2] != other.shape[2]:
        return {}
    heads = ref.shape[2]
    mat = torch.empty(heads, heads, dtype=torch.float32, device=ref.device)
    for i in range(heads):
        av = ref[:, :, i, :].float().reshape(-1)
        av_norm = av.norm()
        for j in range(heads):
            bv = other[:, :, j, :].float().reshape(-1)
            mat[i, j] = (av @ bv) / (av_norm * bv.norm())
    mat_cpu = mat.cpu()
    perm = mat_cpu.argmax(dim=1)
    diag = mat_cpu.diag()
    best = mat_cpu.max(dim=1).values
    return {
        "same_head_matches": int((perm == torch.arange(heads)).sum().item()),
        "diag_cos_mean": float(diag.mean().item()),
        "diag_cos_min": float(diag.min().item()),
        "best_cos_mean": float(best.mean().item()),
        "best_cos_min": float(best.min().item()),
        "perm_first16": perm[: min(16, heads)].tolist(),
    }


def _attention_logits_summary(
    ref_q: torch.Tensor,
    ref_k: torch.Tensor,
    test_q: torch.Tensor,
    test_k: torch.Tensor,
) -> dict[str, Any]:
    """Summarize attention-logit and LSE drift without materializing all heads at once."""
    # Inputs are (B, H, S, D); use batch 0 to keep memory bounded.
    if ref_q.dim() != 4 or ref_k.dim() != 4:
        return {}
    scale = ref_q.shape[-1] ** -0.5
    logits_cos = []
    logits_rel_rmse = []
    lse_cos = []
    lse_rel_rmse = []
    for h in range(ref_q.shape[1]):
        q_ref_h = ref_q[0, h].float()
        k_ref_h = ref_k[0, h].float()
        q_test_h = test_q[0, h].float()
        k_test_h = test_k[0, h].float()
        logits_ref = torch.matmul(q_ref_h, k_ref_h.transpose(-1, -2)) * scale
        logits_test = torch.matmul(q_test_h, k_test_h.transpose(-1, -2)) * scale
        logits_metric = _metric(logits_ref, logits_test)
        lse_ref = torch.logsumexp(logits_ref, dim=-1)
        lse_test = torch.logsumexp(logits_test, dim=-1)
        lse_metric = _metric(lse_ref, lse_test)
        logits_cos.append(logits_metric["cos"])
        logits_rel_rmse.append(logits_metric["rel_rmse_vs_ref_rms"])
        lse_cos.append(lse_metric["cos"])
        lse_rel_rmse.append(lse_metric["rel_rmse_vs_ref_rms"])
    return {
        "logits_cos_mean": float(sum(logits_cos) / len(logits_cos)),
        "logits_cos_min": float(min(logits_cos)),
        "logits_rel_rmse_mean": float(sum(logits_rel_rmse) / len(logits_rel_rmse)),
        "logits_rel_rmse_max": float(max(logits_rel_rmse)),
        "lse_cos_mean": float(sum(lse_cos) / len(lse_cos)),
        "lse_cos_min": float(min(lse_cos)),
        "lse_rel_rmse_mean": float(sum(lse_rel_rmse) / len(lse_rel_rmse)),
        "lse_rel_rmse_max": float(max(lse_rel_rmse)),
    }


def _base_mode(attn_backend: str, ffn_backend: str) -> str:
    active = {attn_backend, ffn_backend}
    if "localcta_fused" in active:
        return "fp4_localcta_fused"
    if "localcta" in active:
        return "fp4_localcta"
    if "tk" in active:
        return "fp4_tk"
    return "fp4_fused_te"


def _configure_backend_pair(attn_backend: str, ffn_backend: str) -> None:
    base_mode = _base_mode(attn_backend, ffn_backend)
    configure_env(base_mode)
    os.environ["FP4_ATTN_BACKEND"] = attn_backend
    os.environ["FP4_FFN_BACKEND"] = ffn_backend
    if "localcta" in {attn_backend, ffn_backend} or "localcta_fused" in {attn_backend, ffn_backend}:
        os.environ["USE_TK_LOCALCTA_DIRECT_CONTRACT"] = "0"
    else:
        os.environ.pop("USE_TK_LOCALCTA_DIRECT_CONTRACT", None)
    _reset_tk_runtime_caches()


def _reset_tk_runtime_caches() -> None:
    import low_bits_training.quantization.tk_gemm as tk_gemm

    tk_gemm._tk_module = None
    tk_gemm._tk_import_attempted = False
    tk_gemm._tk_import_error = None
    tk_gemm._tk_backend_info = {}
    tk_gemm._tk_quant_mod_cache = None
    tk_gemm._tk_localcta_direct_module = None
    tk_gemm._tk_localcta_direct_import_attempted = False
    tk_gemm._tk_localcta_direct_import_error = None


class _FusedAttentionBF16(torch.nn.Module):
    """Exact reference with the same stacked parameter layout as FP4 wrappers."""

    def __init__(self, attention, norm):
        super().__init__()
        self.n_heads = attention.n_heads
        self.n_kv_heads = attention.n_kv_heads
        self.head_dim = attention.head_dim
        self.q_dim = self.n_heads * self.head_dim
        self.k_dim = self.n_kv_heads * self.head_dim
        self.v_dim = self.k_dim
        self.total_out = self.q_dim + self.k_dim + self.v_dim
        self.dim = attention.wq.in_features
        self.epsilon = norm.eps
        self.norm_weight = torch.nn.Parameter(norm.weight.detach().clone())
        self.w_qkv = torch.nn.Parameter(
            torch.cat(
                [attention.wq.weight, attention.wk.weight, attention.wv.weight],
                dim=0,
            ).detach().clone()
        )
        self.wo_weight = torch.nn.Parameter(attention.wo.weight.detach().clone())

    @classmethod
    def from_attention(cls, attention, norm, model_args=None):
        del model_args
        return cls(attention, norm)

    def forward_qkv(self, x, freqs_cis=None, h_carrier=None, cde_row_rms_partial=None):
        del freqs_cis, h_carrier, cde_row_rms_partial
        is_3d = x.dim() == 3
        shape = x.shape
        x_2d = x.reshape(-1, self.dim) if is_3d else x
        normalized = F.rms_norm(
            x_2d,
            (self.dim,),
            self.norm_weight,
            self.epsilon,
        )
        q, k, v = F.linear(normalized, self.w_qkv).split(
            (self.q_dim, self.k_dim, self.v_dim), dim=-1
        )
        self._last_qkv_rope_applied = False
        if is_3d:
            q = q.view(*shape[:-1], self.q_dim)
            k = k.view(*shape[:-1], self.k_dim)
            v = v.view(*shape[:-1], self.v_dim)
        return q, k, v

    def forward_wo(self, attn_output, residual=None, h_gamma=None, cde_emit=False):
        if h_gamma is not None or cde_emit:
            raise RuntimeError("BF16 reference does not accept Wo carriers")
        is_nhsd = attn_output.dim() == 4
        is_3d = attn_output.dim() == 3
        if is_nhsd:
            batch, heads, seq, head_dim = attn_output.shape
            out_2d = attn_output.transpose(1, 2).contiguous().view(
                batch * seq, heads * head_dim
            )
        elif is_3d:
            batch, seq, _ = attn_output.shape
            out_2d = attn_output.reshape(batch * seq, -1)
        else:
            out_2d = attn_output
        output = F.linear(out_2d, self.wo_weight)
        if residual is not None:
            output = output + residual.reshape_as(output)
        if is_nhsd or is_3d:
            output = output.view(batch, seq, self.dim)
        return output


def _make_fused_wrapper(block, fused_cls):
    orig_attn = block.attention
    fused_attn = fused_cls.from_attention(orig_attn, block.attention_norm)
    block.attention = _FusedAttentionWrapper(orig_attn, fused_attn)
    return block


def _build_block_pair(
    flavor: str,
    attn_backend: str,
    ffn_backend: str,
    reference_backend: str,
    device: str,
    seed: int,
):
    _configure_backend_pair(attn_backend, ffn_backend)
    args = _model_args_for_flavor(flavor)

    torch.manual_seed(seed)
    block_base = TransformerBlock(0, args).to(device=device, dtype=torch.bfloat16)
    block_base.init_weights()

    reference_cls = _FusedAttentionBF16 if reference_backend == "bf16" else FusedAttentionFP4_TE
    block_ref = _make_fused_wrapper(copy.deepcopy(block_base), reference_cls)
    fused_cls = FusedAttentionFP4_TE if attn_backend == "te" else FusedAttentionFP4_TK
    block_test = _make_fused_wrapper(copy.deepcopy(block_base), fused_cls)

    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads,
        args.max_seq_len,
        args.rope_theta,
        args.rope_scaling_args,
    ).to(device=device)
    return block_ref, block_test, args, freqs_cis


def _fused_attention_stages(
    block_fused,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    retain_grads: bool = False,
) -> dict[str, torch.Tensor]:
    bs, seqlen, _ = x.shape
    wrapper = block_fused.attention
    fused = wrapper.fused
    q_raw, k_raw, v_raw = fused.forward_qkv(x)
    q = q_raw.view(bs, seqlen, -1, wrapper.head_dim)
    k = k_raw.view(bs, seqlen, -1, wrapper.head_dim)
    v = v_raw.view(bs, seqlen, -1, wrapper.head_dim)
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
    keys = repeat_kv(k_rope, wrapper.n_rep)
    values = repeat_kv(v, wrapper.n_rep)
    q_attn = q_rope.transpose(1, 2)
    k_attn = keys.transpose(1, 2)
    v_attn = values.transpose(1, 2)
    attn_out = wrapper.inner_attention(q_attn, k_attn, v_attn)
    pre_wo = attn_out.transpose(1, 2).contiguous().view(bs, seqlen, -1)
    final = fused.forward_wo(pre_wo)
    stages = {
        "q_raw": q_raw,
        "k_raw": k_raw,
        "v_raw": v_raw,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "q_attn": q_attn,
        "k_attn": k_attn,
        "v_attn": v_attn,
        "pre_wo": pre_wo,
        "final": final,
    }
    if retain_grads:
        for tensor in stages.values():
            tensor.retain_grad()
    return stages


def _split_qkv_grad(stacked: torch.Tensor, q_dim: int, k_dim: int, v_dim: int):
    gw_q, rest = stacked.split([q_dim, k_dim + v_dim], dim=0)
    gw_k, gw_v = rest.split([k_dim, v_dim], dim=0)
    return gw_q, gw_k, gw_v


def _flatten_qkv_cat(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.cat([q, k, v], dim=-1).reshape(-1, q.shape[-1] + k.shape[-1] + v.shape[-1])


def _ones_chunk_grid(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.ones((rows // 128, cols // 128), device=device, dtype=torch.float32)


def _qkv_payload_decomposition(
    block_test,
    x_stage: torch.Tensor,
    ref_stages: dict[str, torch.Tensor],
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    try:
        from low_bits_training.quantization.tk_gemm import _get_tk_quant_for_gemm

        tk_q = _get_tk_quant_for_gemm()
        mod = getattr(tk_q, "_mod", None)
        if mod is None or not hasattr(mod, "tk_localcta_reconstruct_row"):
            return {"error": "localCTA reconstruct helper unavailable"}

        fused = block_test.attention.fused
        x_2d = x_stage.reshape(-1, x_stage.shape[-1]).contiguous()
        normed, inv_rms = ftl._get_te_fused().fused_rmsnorm_only(
            x_2d, fused.norm_weight, float(fused.epsilon)
        )

        shape = payload.get("shape", {})
        M = int(shape.get("M", x_2d.shape[0]))
        K = int(shape.get("K", x_2d.shape[1]))
        N_total = int(shape.get("N_total", fused.total_out))
        mode = payload.get("mode", "unknown")

        if mode == "localcta_direct_debug":
            a_sg = payload["x_sg_direct"]
            b_sg = payload["fwd_b_sg"]
        else:
            a_sg = _ones_chunk_grid(M, K, x_2d.device)
            b_sg = _ones_chunk_grid(N_total, K, x_2d.device)

        x_recon = mod.tk_localcta_reconstruct_row(payload["x_fp4"], payload["x_sc"], a_sg)
        w_recon = mod.tk_localcta_reconstruct_row(payload["wc_fp4_row"], payload["wc_sc_row"], b_sg)
        y_recon = torch.matmul(x_recon.float(), w_recon.float().t()).to(torch.bfloat16)

        ref_y = _flatten_qkv_cat(ref_stages["q_raw"], ref_stages["k_raw"], ref_stages["v_raw"])
        test_y = payload.get("y_cat")
        if test_y is None:
            test_y = torch.cat([payload["xq"], payload["xk"], payload["xv"]], dim=1)
        test_y = test_y.reshape(-1, N_total)

        split_metrics = {}
        offset = 0
        split_widths = payload.get("n_dims", [fused.q_dim, fused.k_dim, fused.v_dim])
        for name, width in zip(("q", "k", "v"), split_widths):
            split_metrics[name] = {
                "recon_vs_ref": _metric(ref_y[:, offset:offset + width], y_recon[:, offset:offset + width]),
                "actual_vs_ref": _metric(ref_y[:, offset:offset + width], test_y[:, offset:offset + width]),
                "actual_vs_recon": _metric(y_recon[:, offset:offset + width], test_y[:, offset:offset + width]),
            }
            offset += width

        return {
            "mode": mode,
            "activation_qdq_vs_normed": _metric(normed, x_recon),
            "weight_qdq_vs_weight": _metric(fused.w_qkv.detach().reshape(N_total, K), w_recon),
            "reconstructed_matmul_vs_ref": _metric(ref_y, y_recon),
            "actual_grouped_gemm_vs_ref": _metric(ref_y, test_y),
            "actual_grouped_gemm_vs_reconstructed_matmul": _metric(y_recon, test_y),
            "per_split": split_metrics,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _qkv_backward_payload_decomposition(
    block_ref,
    block_test,
    x_stage: torch.Tensor,
    x_ref_grad: torch.Tensor,
    fused_bwd_stages: dict[str, torch.Tensor],
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    try:
        from low_bits_training.quantization.tk_gemm import _get_tk_quant_for_gemm

        tk_q = _get_tk_quant_for_gemm()
        mod = getattr(tk_q, "_mod", None)
        if mod is None or not hasattr(mod, "tk_localcta_reconstruct_col"):
            return {"error": "localCTA reconstruct helper unavailable"}

        fused = block_test.attention.fused
        x_2d = x_stage.reshape(-1, x_stage.shape[-1]).contiguous()
        normed, inv_rms = ftl._get_te_fused().fused_rmsnorm_only(
            x_2d, fused.norm_weight, float(fused.epsilon)
        )

        shape = payload.get("shape", {})
        M = int(shape.get("M", x_2d.shape[0]))
        K = int(shape.get("K", x_2d.shape[1]))
        N_total = int(shape.get("N_total", fused.total_out))

        x_col_recon = mod.tk_localcta_reconstruct_col(
            payload["x_fp4_c"],
            payload["x_sc_c"],
            _ones_chunk_grid(K, M, x_2d.device),
        )
        dy_col_recon = mod.tk_localcta_reconstruct_col(
            payload["dy_fp4_cat"],
            payload["dy_sc_cat"],
            _ones_chunk_grid(N_total, M, x_2d.device),
        )
        dW_T_recon = torch.matmul(x_col_recon.float(), dy_col_recon.float().t()).to(torch.bfloat16)
        if payload.get("a_fp4_full") is not None and payload.get("a_sc_cat") is not None:
            dy_row_recon = mod.tk_localcta_reconstruct_row(
                payload["a_fp4_full"],
                payload["a_sc_cat"],
                _ones_chunk_grid(M, N_total, x_2d.device),
            )
        else:
            dy_row_parts = []
            for fp4_i, sc_i in zip(payload.get("fp4_row_list") or [], payload.get("sc_row_list") or []):
                width = fp4_i.shape[1] * 2
                dy_row_parts.append(
                    mod.tk_localcta_reconstruct_row(
                        fp4_i,
                        sc_i,
                        _ones_chunk_grid(M, width, x_2d.device),
                    )
                )
            dy_row_recon = torch.cat(dy_row_parts, dim=1)
        w_col_recon = mod.tk_localcta_reconstruct_col(
            payload["w_fp4_c"],
            payload["w_sc_c"],
            _ones_chunk_grid(K, N_total, x_2d.device),
        )
        dsum_recon = torch.matmul(dy_row_recon.float(), w_col_recon.float().t()).to(torch.bfloat16)
        recon_grad_input, recon_grad_gamma = ftl._te_ffn_rmsnorm_backward_reference(
            dsum_recon,
            x_2d,
            fused.norm_weight,
            inv_rms,
        )
        local_dy = _flatten_qkv_cat(
            fused_bwd_stages["q_raw"].grad,
            fused_bwd_stages["k_raw"].grad,
            fused_bwd_stages["v_raw"].grad,
        ).t().contiguous()
        ref_dW_T = block_ref.attention.fused.w_qkv.grad.t().contiguous()
        actual_dW_T = payload["dW_T"]
        return {
            "mode": payload.get("mode", "unknown"),
            "x_col_qdq_vs_normed_t": _metric(normed.t().contiguous(), x_col_recon),
            "dy_col_qdq_vs_local_grad_t": _metric(local_dy, dy_col_recon),
            "dy_row_qdq_vs_local_grad": _metric(local_dy.t().contiguous(), dy_row_recon),
            "reconstructed_wgrad_vs_te_ref": _metric(ref_dW_T, dW_T_recon),
            "actual_wgrad_vs_te_ref": _metric(ref_dW_T, actual_dW_T),
            "actual_wgrad_vs_reconstructed_wgrad": _metric(dW_T_recon, actual_dW_T),
            "reconstructed_dgrad_vs_actual_dsum": _metric(dsum_recon, payload["D_sum"]),
            "reconstructed_dgrad_vs_reconstructed_rmsnorm_input": _metric(recon_grad_input, payload["grad_input"]),
            "actual_rmsnorm_grad_input_vs_te_ref": _metric(x_ref_grad, payload["grad_input"]),
            "reconstructed_rmsnorm_grad_input_vs_te_ref": _metric(x_ref_grad, recon_grad_input),
            "actual_rmsnorm_grad_gamma_vs_te_ref": _metric(
                block_ref.attention.fused.norm_weight.grad,
                payload["grad_norm_weight"],
            ),
            "reconstructed_rmsnorm_grad_gamma_vs_te_ref": _metric(
                block_ref.attention.fused.norm_weight.grad,
                recon_grad_gamma.to(block_ref.attention.fused.norm_weight.grad.dtype),
            ),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_attention_parity(
    flavor: str,
    batch_size: int,
    seq_len: int,
    attn_backend: str,
    ffn_backend: str,
    reference_backend: str,
    device: str,
    seed: int,
    *,
    qkv_decompose: bool = False,
) -> dict[str, Any]:
    prev_cat_debug = os.environ.get("USE_TK_QKV_FORWARD_CAT_DEBUG")
    prev_bwd_debug = os.environ.get("USE_TK_QKV_BWD_CAPTURE_DEBUG")
    if qkv_decompose and attn_backend.startswith("localcta"):
        os.environ["USE_TK_QKV_FORWARD_CAT_DEBUG"] = "1"
        os.environ["USE_TK_QKV_BWD_CAPTURE_DEBUG"] = "1"
    try:
        block_ref, block_test, model_args, freqs_cis = _build_block_pair(
            flavor, attn_backend, ffn_backend, reference_backend, device, seed
        )

        torch.manual_seed(seed + 1)
        x_stage = torch.randn(batch_size, seq_len, model_args.dim, device=device, dtype=torch.bfloat16)
        ref_stages = _fused_attention_stages(block_ref, x_stage, freqs_cis)
        if qkv_decompose and attn_backend.startswith("localcta"):
            ftl._get_last_qkv_forward_debug_payload(clear=True)
        fused_stages = _fused_attention_stages(block_test, x_stage, freqs_cis)
        qkv_payload = (
            ftl._get_last_qkv_forward_debug_payload(clear=True)
            if qkv_decompose and attn_backend.startswith("localcta")
            else None
        )
    finally:
        if prev_cat_debug is None:
            os.environ.pop("USE_TK_QKV_FORWARD_CAT_DEBUG", None)
        else:
            os.environ["USE_TK_QKV_FORWARD_CAT_DEBUG"] = prev_cat_debug

    stage_metrics = {name: _metric(ref_stages[name], fused_stages[name]) for name in ref_stages}
    decomposition = _qkv_payload_decomposition(block_test, x_stage, ref_stages, qkv_payload)

    q_dim = block_test.attention.fused.q_dim
    k_dim = block_test.attention.fused.k_dim
    v_dim = block_test.attention.fused.v_dim

    backward_metrics = None
    backward_error = None
    backward_finite = False
    backward_decomposition = None
    if qkv_decompose and attn_backend.startswith("localcta"):
        os.environ["USE_TK_QKV_BWD_CAPTURE_DEBUG"] = "1"
    try:
        x_ref = x_stage.detach().clone().requires_grad_(True)
        x_fused = x_stage.detach().clone().requires_grad_(True)
        gout = torch.randn(batch_size, seq_len, model_args.dim, device=device, dtype=torch.bfloat16)

        block_ref.zero_grad(set_to_none=True)
        block_test.zero_grad(set_to_none=True)
        ref_bwd_stages = _fused_attention_stages(block_ref, x_ref, freqs_cis, retain_grads=True)
        fused_bwd_stages = _fused_attention_stages(block_test, x_fused, freqs_cis, retain_grads=True)
        y_ref = ref_bwd_stages["final"]
        y_fused = fused_bwd_stages["final"]
        if qkv_decompose and attn_backend.startswith("localcta"):
            from low_bits_training.quantization import tk_gemm as tk_gemm
            tk_gemm._get_last_qkv_backward_debug_payload(clear=True)
        torch.autograd.backward(y_ref, gout)
        torch.autograd.backward(y_fused, gout)
        qkv_backward_payload = (
            tk_gemm._get_last_qkv_backward_debug_payload(clear=True)
            if qkv_decompose and attn_backend.startswith("localcta")
            else None
        )

        ref_gw_q, ref_gw_k, ref_gw_v = _split_qkv_grad(block_ref.attention.fused.w_qkv.grad, q_dim, k_dim, v_dim)
        gw_q, gw_k, gw_v = _split_qkv_grad(block_test.attention.fused.w_qkv.grad, q_dim, k_dim, v_dim)
        stage_backward_metrics = {
            "grad_final": _metric(y_ref.grad, y_fused.grad),
            "grad_pre_wo": _metric(ref_bwd_stages["pre_wo"].grad, fused_bwd_stages["pre_wo"].grad),
            "grad_q_attn": _metric(ref_bwd_stages["q_attn"].grad, fused_bwd_stages["q_attn"].grad),
            "grad_k_attn": _metric(ref_bwd_stages["k_attn"].grad, fused_bwd_stages["k_attn"].grad),
            "grad_v_attn": _metric(ref_bwd_stages["v_attn"].grad, fused_bwd_stages["v_attn"].grad),
            "grad_q_rope": _metric(ref_bwd_stages["q_rope"].grad, fused_bwd_stages["q_rope"].grad),
            "grad_k_rope": _metric(ref_bwd_stages["k_rope"].grad, fused_bwd_stages["k_rope"].grad),
            "grad_q_raw": _metric(ref_bwd_stages["q_raw"].grad, fused_bwd_stages["q_raw"].grad),
            "grad_k_raw": _metric(ref_bwd_stages["k_raw"].grad, fused_bwd_stages["k_raw"].grad),
            "grad_v_raw": _metric(ref_bwd_stages["v_raw"].grad, fused_bwd_stages["v_raw"].grad),
        }
        backward_stage_order = [
            "grad_final",
            "grad_pre_wo",
            "grad_q_attn",
            "grad_k_attn",
            "grad_v_attn",
            "grad_q_rope",
            "grad_k_rope",
            "grad_q_raw",
            "grad_k_raw",
            "grad_v_raw",
            "grad_x",
            "grad_w_q",
            "grad_w_k",
            "grad_w_v",
            "grad_w_qkv",
            "grad_w_wo",
            "grad_norm_weight",
        ]
        backward_metrics = {
            "grad_x": _metric(x_ref.grad, x_fused.grad),
            "grad_w_q": _metric(ref_gw_q, gw_q),
            "grad_w_k": _metric(ref_gw_k, gw_k),
            "grad_w_v": _metric(ref_gw_v, gw_v),
            "grad_w_qkv": _metric(block_ref.attention.fused.w_qkv.grad, block_test.attention.fused.w_qkv.grad),
            "grad_w_wo": _metric(block_ref.attention.fused.wo_weight.grad, block_test.attention.fused.wo_weight.grad),
            "grad_norm_weight": _metric(block_ref.attention.fused.norm_weight.grad, block_test.attention.fused.norm_weight.grad),
            **stage_backward_metrics,
            "stage_grads": stage_backward_metrics,
            "summary": _summarize_metric_chain(
                {
                    **stage_backward_metrics,
                    "grad_x": _metric(x_ref.grad, x_fused.grad),
                    "grad_w_q": _metric(ref_gw_q, gw_q),
                    "grad_w_k": _metric(ref_gw_k, gw_k),
                    "grad_w_v": _metric(ref_gw_v, gw_v),
                    "grad_w_qkv": _metric(block_ref.attention.fused.w_qkv.grad, block_test.attention.fused.w_qkv.grad),
                    "grad_w_wo": _metric(block_ref.attention.fused.wo_weight.grad, block_test.attention.fused.wo_weight.grad),
                    "grad_norm_weight": _metric(block_ref.attention.fused.norm_weight.grad, block_test.attention.fused.norm_weight.grad),
                },
                backward_stage_order,
            ),
        }
        backward_finite = all(
            bool(torch.isfinite(t).all().item())
            for t in (
                x_ref.grad,
                x_fused.grad,
                block_ref.attention.fused.w_qkv.grad,
                block_ref.attention.fused.wo_weight.grad,
                block_ref.attention.fused.norm_weight.grad,
                block_test.attention.fused.w_qkv.grad,
                block_test.attention.fused.wo_weight.grad,
                block_test.attention.fused.norm_weight.grad,
            )
        )
        backward_decomposition = _qkv_backward_payload_decomposition(
            block_ref,
            block_test,
            x_stage,
            x_ref.grad,
            fused_bwd_stages,
            qkv_backward_payload,
        )
    except Exception as exc:
        backward_error = f"{type(exc).__name__}: {exc}"
    finally:
        if prev_bwd_debug is None:
            os.environ.pop("USE_TK_QKV_BWD_CAPTURE_DEBUG", None)
        else:
            os.environ["USE_TK_QKV_BWD_CAPTURE_DEBUG"] = prev_bwd_debug

    return {
        "flavor": flavor,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "attn_backend": attn_backend,
        "ffn_backend": ffn_backend,
        "reference_backend": reference_backend,
        "shape": {
            "dim": model_args.dim,
            "q_dim": q_dim,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "head_dim": block_ref.attention.head_dim,
        },
        "stages": stage_metrics,
        "structure": {
            "q_rope_head_alignment": _head_alignment_summary(ref_stages["q_rope"], fused_stages["q_rope"]),
            "k_rope_head_alignment": _head_alignment_summary(ref_stages["k_rope"], fused_stages["k_rope"]),
            "attention_logits": _attention_logits_summary(
                ref_stages["q_attn"],
                ref_stages["k_attn"],
                fused_stages["q_attn"],
                fused_stages["k_attn"],
            ),
        },
        "backward": backward_metrics,
        "backward_error": backward_error,
        "qkv_decomposition": decomposition,
        "qkv_backward_decomposition": backward_decomposition,
        "finite": {
            "forward": all(bool(torch.isfinite(t).all().item()) for t in ref_stages.values())
            and all(bool(torch.isfinite(t).all().item()) for t in fused_stages.values()),
            "backward": backward_finite,
        },
    }


def main():
    args = parse_args()
    try:
        _preinit_cuda(args.device_index)
    except Exception:
        torch.cuda.set_device(args.device_index)
        torch.empty(1, device=f"cuda:{args.device_index}")
        torch.cuda.synchronize()
    _set_device_with_retry(args.device_index)
    device = f"cuda:{args.device_index}"

    results = []
    for combo in args.backend_combos:
        attn_backend, ffn_backend = combo.split("/", 1)
        if attn_backend not in VALID_BACKENDS or ffn_backend not in VALID_BACKENDS:
            raise ValueError(f"invalid backend combo {combo!r}")
        result = run_attention_parity(
            args.flavor,
            args.batch_size,
            args.seq_len,
            attn_backend,
            ffn_backend,
            args.reference_backend,
            device,
            args.seed,
            qkv_decompose=args.qkv_decompose,
        )
        results.append(result)
        backward = result["backward"]
        decomposition = result.get("qkv_decomposition")
        if backward is None:
            print(
                f"{combo} final={result['stages']['final']['cos']:.8f} "
                f"pre_wo={result['stages']['pre_wo']['cos']:.8f} "
                f"backward_error={result['backward_error']}",
                flush=True,
            )
        else:
            summary = backward.get("summary", {})
            print(
                f"{combo} final={result['stages']['final']['cos']:.8f} "
                f"pre_wo={result['stages']['pre_wo']['cos']:.8f} "
                f"grad_x={backward['grad_x']['cos']:.8f} "
                f"grad_w_q={backward['grad_w_q']['cos']:.8f} "
                f"grad_w_wo={backward['grad_w_wo']['cos']:.8f} "
                f"first_catastrophic={summary.get('first_catastrophic_tensor')} "
                f"first_nonfinite={summary.get('first_nonfinite_tensor')}",
                flush=True,
            )
        if decomposition is not None and "error" not in decomposition:
            print(
                f"{combo} qdq_act={decomposition['activation_qdq_vs_normed']['cos']:.8f} "
                f"qdq_w={decomposition['weight_qdq_vs_weight']['cos']:.8f} "
                f"recon_vs_ref={decomposition['reconstructed_matmul_vs_ref']['cos']:.8f} "
                f"actual_vs_ref={decomposition['actual_grouped_gemm_vs_ref']['cos']:.8f} "
                f"actual_vs_recon={decomposition['actual_grouped_gemm_vs_reconstructed_matmul']['cos']:.8f}",
                flush=True,
            )
        elif decomposition is not None:
            print(f"{combo} qkv_decomposition_error={decomposition['error']}", flush=True)
        backward_decomposition = result.get("qkv_backward_decomposition")
        if backward_decomposition is not None and "error" not in backward_decomposition:
            print(
                f"{combo} qdq_xcol={backward_decomposition['x_col_qdq_vs_normed_t']['cos']:.8f} "
                f"qdq_dycol={backward_decomposition['dy_col_qdq_vs_local_grad_t']['cos']:.8f} "
                f"qdq_dyrow={backward_decomposition['dy_row_qdq_vs_local_grad']['cos']:.8f} "
                f"wgrad_recon_vs_te={backward_decomposition['reconstructed_wgrad_vs_te_ref']['cos']:.8f} "
                f"wgrad_actual_vs_te={backward_decomposition['actual_wgrad_vs_te_ref']['cos']:.8f} "
                f"wgrad_actual_vs_recon={backward_decomposition['actual_wgrad_vs_reconstructed_wgrad']['cos']:.8f} "
                f"dgrad_actual_vs_recon={backward_decomposition['reconstructed_dgrad_vs_actual_dsum']['cos']:.8f} "
                f"rms_in_actual_vs_te={backward_decomposition['actual_rmsnorm_grad_input_vs_te_ref']['cos']:.8f}",
                flush=True,
            )
        elif backward_decomposition is not None:
            print(f"{combo} qkv_backward_decomposition_error={backward_decomposition['error']}", flush=True)

    report = {
        "harness": "compare_fp4_attention_wrapper_numerics",
        "device_index": args.device_index,
        "results": results,
    }
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote attention parity JSON to {args.json_out}", flush=True)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
