"""
Full model comparison: Working (TritonTEParityLinear) vs V2 Fused (TritonFusedQuantLinear)
With torch.compile and multiple training iterations on a small Llama model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
import numpy as np
import csv

# Must be set before importing model
os.environ["NVTE_NVFP4_DISABLE_RHT"] = "1"
os.environ["NVTE_NVFP4_DISABLE_2D_QUANTIZATION"] = "0"
os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = "0"
os.environ["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] = "1"

from low_bits_training.models.llama3 import Transformer, TransformerModelArgs
from low_bits_training.quantization.te_parity_linear_triton import TritonTEParityLinear
from low_bits_training.quantization.fused_quant_triton_v2 import TritonFusedQuantLinear
from low_bits_training.quantization.te_parity_linear_tex import TEParityLinearTex
from low_bits_training.quantization.mxfp_custom_te_fp4 import (
    BoundRecipeLinear,
    NVFP4BlockScaling,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class MockCfg:
    block_size = 16
    scale_type = "E4M3"
    use_global_scale = True
    encode_centric = False
    strategy = "decode"
    use_rht = False
    scale_round_mode = "TiesToEven"
    roundMode = "TiesToEven"
    use_2d_weights = True
    use_bf16_data = True
    use_fp32_matmul = True
    scale_max = 448.0


def replace_linear_with_working(module, mx_config, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "output":
            continue
        if isinstance(child, nn.Linear):
            custom = TritonTEParityLinear.from_float(child, mx_config)
            custom.name = full_name
            setattr(module, name, custom)
        else:
            replace_linear_with_working(child, mx_config, full_name)


def replace_linear_with_v2(module, mx_config, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "output":
            continue
        if isinstance(child, nn.Linear):
            v2_layer = TritonFusedQuantLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                mx_config=mx_config,
            )
            v2_layer = v2_layer.to(child.weight.dtype)
            with torch.no_grad():
                v2_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    v2_layer.bias.copy_(child.bias)
            v2_layer.name = full_name
            setattr(module, name, v2_layer)
        else:
            replace_linear_with_v2(child, mx_config, full_name)


def replace_linear_with_tex(module, mx_config, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "output":
            continue
        if isinstance(child, nn.Linear):
            tex_layer = TEParityLinearTex(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                mx_config=mx_config,
                use_dequant_gemm=False,
            )
            tex_layer = tex_layer.to(child.weight.dtype)
            with torch.no_grad():
                tex_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    tex_layer.bias.copy_(child.bias)
            tex_layer.name = full_name
            setattr(module, name, tex_layer)
        else:
            replace_linear_with_tex(child, mx_config, full_name)


def replace_linear_with_mxfp(module, recipe, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name == "output":
            continue
        if isinstance(child, nn.Linear):
            te_layer = BoundRecipeLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
                params_dtype=child.weight.dtype,
                recipe=recipe,
                device=child.weight.device,
            )
            with torch.no_grad():
                te_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    te_layer.bias.copy_(child.bias)
            te_layer.name = full_name
            setattr(module, name, te_layer)
        else:
            replace_linear_with_mxfp(child, recipe, full_name)


def copy_weights(model_src, model_dst):
    sd_src = model_src.state_dict()
    sd_dst = model_dst.state_dict()
    for key in sd_dst.keys():
        if key in sd_src and sd_dst[key].shape == sd_src[key].shape:
            sd_dst[key].copy_(sd_src[key])
    model_dst.load_state_dict(sd_dst)


def compare_state_dicts(model_a, model_b):
    sd_a = {
        k: v.float().cpu() for k, v in model_a.state_dict().items() if "scaling" not in k
    }
    sd_b = {
        k: v.float().cpu() for k, v in model_b.state_dict().items() if "scaling" not in k
    }
    max_diff, violator = 0.0, ""
    for k in sd_a.keys():
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            diff = (sd_a[k] - sd_b[k]).abs().max().item()
            if diff > max_diff:
                max_diff, violator = diff, k
    return max_diff, violator


def compare_grads(model_a, model_b):
    sd_a = {
        k: v.grad.float().cpu()
        for k, v in model_a.named_parameters()
        if v.grad is not None and "scaling" not in k
    }
    sd_b = {
        k: v.grad.float().cpu()
        for k, v in model_b.named_parameters()
        if v.grad is not None and "scaling" not in k
    }
    max_diff, violator = 0.0, ""
    for k in sd_a.keys():
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            diff = (sd_a[k] - sd_b[k]).abs().max().item()
            if diff > max_diff:
                max_diff, violator = diff, k
    return max_diff, violator


def run_comparison(num_steps=50, use_compile=True, model_size="small"):
    print("=" * 205)
    print(f"Full Model Comparison: Working vs V2 vs TEX Native vs MXFP TE")
    print(f"Steps: {num_steps}, torch.compile: {use_compile}, Model: {model_size}")
    print("=" * 205)

    set_seed(42)
    device, dtype = "cuda", torch.bfloat16

    if model_size == "tiny":
        model_args = TransformerModelArgs(
            dim=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            ffn_dim_multiplier=1.0,
            multiple_of=64,
            rope_theta=10000,
            vocab_size=1024,
        )
        batch_size, seq_len = 4, 128
    elif model_size == "small":
        model_args = TransformerModelArgs(
            dim=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,
            ffn_dim_multiplier=1.0,
            multiple_of=128,
            rope_theta=10000,
            vocab_size=8000,
        )
        batch_size, seq_len = 2, 256
    else:
        model_args = TransformerModelArgs(
            dim=2048,
            n_layers=24,
            n_heads=32,
            n_kv_heads=32,
            ffn_dim_multiplier=1.0,
            multiple_of=256,
            rope_theta=10000,
            vocab_size=32000,
        )
        batch_size, seq_len = 2, 512

    mx_config = MockCfg()
    base_model = Transformer(model_args).to(dtype)

    model_working = copy.deepcopy(base_model)
    replace_linear_with_working(model_working, mx_config)
    model_working = model_working.to(device)

    model_v2 = copy.deepcopy(base_model)
    replace_linear_with_v2(model_v2, mx_config)
    model_v2 = model_v2.to(device)

    model_tex = copy.deepcopy(base_model)
    replace_linear_with_tex(model_tex, mx_config)
    model_tex = model_tex.to(device)

    model_mxfp = copy.deepcopy(base_model)
    replace_linear_with_mxfp(model_mxfp, NVFP4BlockScaling())
    model_mxfp = model_mxfp.to(device)

    copy_weights(model_working, model_v2)
    copy_weights(model_working, model_tex)
    copy_weights(model_working, model_mxfp)

    # dY Extraction Setup
    dy_storage_working = {}
    dy_storage_v2 = {}
    dy_storage_tex = {}
    dy_storage_mxfp = {}

    def get_dy_hook(name, storage):
        def hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0:
                storage[name] = grad_output[0].detach()

        return hook

    def register_hooks(model, storage):
        for name, m in model.named_modules():
            if isinstance(
                m,
                (
                    TritonTEParityLinear,
                    TritonFusedQuantLinear,
                    TEParityLinearTex,
                    BoundRecipeLinear,
                ),
            ):
                m.register_full_backward_hook(get_dy_hook(name, storage))

    register_hooks(model_working, dy_storage_working)
    register_hooks(model_v2, dy_storage_v2)
    register_hooks(model_tex, dy_storage_tex)
    register_hooks(model_mxfp, dy_storage_mxfp)

    if use_compile:
        model_working = torch.compile(model_working)
        model_v2 = torch.compile(model_v2)
        model_tex = torch.compile(model_tex)
        model_mxfp = torch.compile(model_mxfp)

    lr = 1e-3
    opt_working = optim.AdamW(model_working.parameters(), lr=lr)
    opt_v2 = optim.AdamW(model_v2.parameters(), lr=lr)
    opt_tex = optim.AdamW(model_tex.parameters(), lr=lr)
    opt_mxfp = optim.AdamW(model_mxfp.parameters(), lr=lr)

    print(
        f"{'Step':>5} | {'Loss W':>10} | {'Loss V':>10} | {'Loss T':>10} | {'Loss M':>10} | "
        f"{'L(W-V)':>10} | {'L(W-T)':>10} | {'L(W-M)':>10} | {'L(V-T)':>10} | {'L(V-M)':>10} | {'L(T-M)':>10} | "
        f"{'G(W-V)':>10} | {'G(W-T)':>10} | {'G(W-M)':>10} | {'G(V-T)':>10} | {'G(V-M)':>10} | {'G(T-M)':>10}"
    )
    print("-" * 255)

    csv_filename = "comparison_results.csv"
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "step",
            "loss_w",
            "loss_v",
            "loss_t",
            "loss_m",
            "L_wv",
            "L_wt",
            "L_wm",
            "L_vt",
            "L_vm",
            "L_tm",
            "G_wv",
            "G_wt",
            "G_wm",
            "G_vt",
            "G_vm",
            "G_tm",
            "P_wv",
            "P_wt",
            "P_wm",
            "P_vt",
            "P_vm",
            "P_tm",
            "dY_wv",
            "dY_wt",
            "dY_wm",
            "dY_vt",
            "dY_vm",
            "dY_tm",
        ]
    )

    for step in range(1, num_steps + 1):
        set_seed(42 + step)
        tokens = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )
        targets = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )

        # Clear storage
        dy_storage_working.clear()
        dy_storage_working.clear()
        dy_storage_v2.clear()
        dy_storage_tex.clear()
        dy_storage_mxfp.clear()

        # Step Working
        opt_working.zero_grad()
        loss_working = F.cross_entropy(
            model_working(tokens).view(-1, model_args.vocab_size), targets.view(-1)
        )
        loss_working.backward()

        # Step V2
        opt_v2.zero_grad()
        loss_v2 = F.cross_entropy(
            model_v2(tokens).view(-1, model_args.vocab_size), targets.view(-1)
        )
        loss_v2.backward()

        # Step TEX
        opt_tex.zero_grad()
        loss_tex = F.cross_entropy(
            model_tex(tokens).view(-1, model_args.vocab_size), targets.view(-1)
        )
        loss_tex.backward()

        # Step MXFP
        opt_mxfp.zero_grad()
        loss_mxfp = F.cross_entropy(
            model_mxfp(tokens).view(-1, model_args.vocab_size), targets.view(-1)
        )
        loss_mxfp.backward()

        # Extract Raw Models for robust comparisons (handle compiled vs uncompiled)
        # Note: We use these for grad/param structural comparisons to ensure keys match
        m_w = (
            model_working._orig_mod
            if hasattr(model_working, "_orig_mod")
            else model_working
        )
        m_v = model_v2._orig_mod if hasattr(model_v2, "_orig_mod") else model_v2
        m_t = model_tex._orig_mod if hasattr(model_tex, "_orig_mod") else model_tex
        m_m = model_mxfp._orig_mod if hasattr(model_mxfp, "_orig_mod") else model_mxfp

        # Comparisons (WV, WT, WM, VT, VM, TM)
        grad_diff_wv, _ = compare_grads(m_w, m_v)
        grad_diff_wt, _ = compare_grads(m_w, m_t)
        grad_diff_wm, _ = compare_grads(m_w, m_m)
        grad_diff_vt, _ = compare_grads(m_v, m_t)
        grad_diff_vm, _ = compare_grads(m_v, m_m)
        grad_diff_tm, _ = compare_grads(m_t, m_m)

        def get_dy_diff(s1, s2):
            diff = 0.0
            for name in s1:
                if name in s2:
                    val = (s1[name] - s2[name]).abs().max().item()
                    diff = max(diff, val)
            return diff

        dy_diff_wv = get_dy_diff(dy_storage_working, dy_storage_v2)
        dy_diff_wt = get_dy_diff(dy_storage_working, dy_storage_tex)
        dy_diff_wm = get_dy_diff(dy_storage_working, dy_storage_mxfp)
        dy_diff_vt = get_dy_diff(dy_storage_v2, dy_storage_tex)
        dy_diff_vm = get_dy_diff(dy_storage_v2, dy_storage_mxfp)
        dy_diff_tm = get_dy_diff(dy_storage_tex, dy_storage_mxfp)

        loss_diff_wv = abs(loss_working.item() - loss_v2.item())
        loss_diff_wt = abs(loss_working.item() - loss_tex.item())
        loss_diff_wm = abs(loss_working.item() - loss_mxfp.item())
        loss_diff_vt = abs(loss_v2.item() - loss_tex.item())
        loss_diff_vm = abs(loss_v2.item() - loss_mxfp.item())
        loss_diff_tm = abs(loss_tex.item() - loss_mxfp.item())

        opt_working.step()
        opt_v2.step()
        opt_tex.step()
        opt_mxfp.step()

        param_diff_wv, _ = compare_state_dicts(m_w, m_v)
        param_diff_wt, _ = compare_state_dicts(m_w, m_t)
        param_diff_wm, _ = compare_state_dicts(m_w, m_m)
        param_diff_vt, _ = compare_state_dicts(m_v, m_t)
        param_diff_vm, _ = compare_state_dicts(m_v, m_m)
        param_diff_tm, _ = compare_state_dicts(m_t, m_m)

        if (
            step <= 5
            or step % 10 == 0
            or max(loss_diff_wv, loss_diff_wt, loss_diff_wm) > 0.01
            or max(param_diff_wv, param_diff_wt, param_diff_wm) > 0.01
        ):
            print(
                f"{step:>5} | {loss_working.item():>10.4f} | {loss_v2.item():>10.4f} | {loss_tex.item():>10.4f} | {loss_mxfp.item():>10.4f} | "
                f"{loss_diff_wv:>10.2e} | {loss_diff_wt:>10.2e} | {loss_diff_wm:>10.2e} | "
                f"{loss_diff_vt:>10.2e} | {loss_diff_vm:>10.2e} | {loss_diff_tm:>10.2e} | "
                f"{grad_diff_wv:>10.2e} | {grad_diff_wt:>10.2e} | {grad_diff_wm:>10.2e} | "
                f"{grad_diff_vt:>10.2e} | {grad_diff_vm:>10.2e} | {grad_diff_tm:>10.2e}"
            )

        csv_writer.writerow(
            [
                step,
                loss_working.item(),
                loss_v2.item(),
                loss_tex.item(),
                loss_mxfp.item(),
                loss_diff_wv,
                loss_diff_wt,
                loss_diff_wm,
                loss_diff_vt,
                loss_diff_vm,
                loss_diff_tm,
                grad_diff_wv,
                grad_diff_wt,
                grad_diff_wm,
                grad_diff_vt,
                grad_diff_vm,
                grad_diff_tm,
                param_diff_wv,
                param_diff_wt,
                param_diff_wm,
                param_diff_vt,
                param_diff_vm,
                param_diff_tm,
                dy_diff_wv,
                dy_diff_wt,
                dy_diff_wm,
                dy_diff_vt,
                dy_diff_vm,
                dy_diff_tm,
            ]
        )
        csv_file.flush()

    csv_file.close()
    print(f"\n✅ Run finished. Results saved to {csv_filename}")


if __name__ == "__main__":
    run_comparison()
