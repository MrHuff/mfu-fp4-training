import os

# # Env vars (Same as debug_llama_parity)
# os.environ["NVTE_NVFP4_DISABLE_RHT"] = "1"
# os.environ["NVTE_NVFP4_DISABLE_2D_QUANTIZATION"] = "0"
# os.environ["NVTE_NVFP4_ENCODE_CENTRIC"] = "0"
# os.environ["NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
import numpy as np
import sys

# Helper paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "tests"))

# Import Debug Tools
from low_bits_training.quantization.debug_fused_quant import debug_fake_quant_simultaneous
from low_bits_training.quantization.fused_quant_triton_v2 import TritonFusedQuantLinear
from low_bits_training.quantization.te_parity_linear_triton import TritonTEParityLinear
from low_bits_training.quantization.te_parity_linear_tex import (
    TEParityLinearTex,
    TEParityLinearTexFunction,
)
from low_bits_training.models.llama3 import Transformer, TransformerModelArgs
from transformer_engine.pytorch import NVFP4Quantizer

# Reuse Helpers from debug_llama_parity
from debug_llama_parity import MockCfg, replace_linear, compare, get_param


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


# QGEMM Imports
try:
    from transformer_engine.pytorch.experimental.quantization_custom_triton import (
        QuantizationMetadata,
        get_triton_quantizer_factory,
    )

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("WARNING: TE not available")

# ----------------- MONKEY PATCHED ABLATION LAYER -----------------


# Import fused module for patching
import low_bits_training.quantization.fused_quant_triton_v2

# Global Capture Lists (OUR OWN - don't import from debug_llama_parity to avoid double-capture)
# Global Capture Lists (OUR OWN - don't import from debug_llama_parity to avoid double-capture)
CAPTURED_ABLATION = {}
CAPTURED_REF = {}

# Context for identifying current layer
CURRENT_LAYER_NAME = "Unknown"
CURRENT_PASS = "FWD"
CAPTURE_COUNTERS = {}  # (layer_name, pass_type) -> count


# Import TritonQuantizer to patch
from transformer_engine.pytorch.experimental.quantization_custom_triton import (
    get_triton_quantizer_factory,
)

# Get Class dynamically
_dummy_factory = get_triton_quantizer_factory()
TritonQuantizer = type(_dummy_factory("dummy"))


# Monkey Patch Wrapper for Reference Quantizer
def debug_quantize_ref_wrapper(
    self, x, enable_rht=None, scale_dtype=torch.float32, data_dtype=torch.float32
):
    # Compute expected AMAX from input before any internal transforms
    input_amax = x.float().abs().max().item()

    # Check RHT status on first call
    if len(CAPTURED_REF) == 0:
        print(
            f"[REF] Quantizer config: with_rht={getattr(self, 'with_rht', 'Unknown')}, block_size={getattr(self, 'block_size', 'Unknown')}"
        )

    # Call original - MUST pass enable_rht to match signature!
    q_x = TritonQuantizer._original_quantize_rowcol_v2(
        self, x, enable_rht, scale_dtype, data_dtype
    )

    # Capture stats
    inp_mean = x.float().mean().item()
    inp_absmax = x.abs().max().item()

    # Capture (Detached) - include both row and col data for proper comparison
    capture_data = {
        "type": "Ref_Quant",
        "input_orig": x.detach(),  # Original input for verification
        "inp_mean": inp_mean,  # NEW
        "inp_absmax": inp_absmax,  # NEW
        "input_amax_direct": input_amax,  # What we compute from raw input
        "input_shape": x.shape,
        "q_data_row": q_x.data_row.detach(),
        "q_scale_row": q_x.scale_row.detach(),
        "q_data_col": q_x.data_col.detach() if hasattr(q_x, "data_col") else None,
        "q_scale_col": q_x.scale_col.detach() if hasattr(q_x, "scale_col") else None,
        "global_amax": q_x.global_amax.detach() if hasattr(q_x, "global_amax") else None,
        "layer": CURRENT_LAYER_NAME,
        "pass": CURRENT_PASS,
    }

    if CURRENT_LAYER_NAME not in CAPTURED_REF:
        CAPTURED_REF[CURRENT_LAYER_NAME] = {}

    # Generate Key
    counter_key = (CURRENT_LAYER_NAME, CURRENT_PASS, "Ref")
    count = CAPTURE_COUNTERS.get(counter_key, 0)
    CAPTURE_COUNTERS[counter_key] = count + 1
    capture_key = f"{CURRENT_PASS}_{count}"

    CAPTURED_REF[CURRENT_LAYER_NAME][capture_key] = capture_data
    return q_x


# Save original
if not hasattr(TritonQuantizer, "_original_quantize_rowcol_v2"):
    TritonQuantizer._original_quantize_rowcol_v2 = TritonQuantizer.quantize_rowcol_v2

# Apply Patch
# Apply Patch
TritonQuantizer.quantize_rowcol_v2 = debug_quantize_ref_wrapper


# Monkey Patch Wrapper for TEX Quantizer
def debug_quantize_tex_wrapper(self, tensor):
    # Capture input stats
    # Compute expected AMAX from input before any internal transforms
    input_amax = tensor.float().abs().max().item()
    inp_mean = tensor.float().mean().item()
    inp_absmax = tensor.abs().max().item()

    # Call original
    q_out = NVFP4Quantizer._original_quantize(self, tensor)

    # Decode Data and Scales for Comparison
    # We use TEParityLinearTexFunction internals to unpack

    # 1. Rowwise
    q_data_row = None
    q_scale_row = None
    if q_out._rowwise_data is not None:
        # Unpack data similar to _manual_dequantize but without scaling
        # We need to replicate the unpacking logic or use a helper if we refactored it separate from dequant
        # Since _manual_dequantize does everything, let's just use the first part logic here inline or call the helper
        # But the helper returns (data * scale). We want separated.

        # Unpack Data
        R, C = q_out.shape
        data_packed = q_out._rowwise_data
        scale_inv = q_out._rowwise_scale_inv

        # Unpack logic from te_parity_linear_tex.py
        data_u8 = data_packed.view(torch.uint8).to(torch.int32)
        unpacked_indices = torch.stack((data_u8 & 0x0F, data_u8 >> 4), dim=-1).reshape(
            R, C
        )
        lut = TEParityLinearTexFunction._fp4_e2m1_vals(tensor.device, torch.float32)
        q_data_row = lut[unpacked_indices.to(torch.long)]

        # Decode Scale
        if scale_inv is not None:
            # REF returns (R, C//BS).
            # Fused Scale X (s_a) matches (M, K//BS).
            # So Ref Row Scale matches Fused Row Scale directly.
            q_scale_row = TEParityLinearTexFunction._decode_scale(
                scale_inv, (R, C), "E4M3", expand=False
            )
            # Experiment: Apply 448 factor if Ref is scale_inv (small) vs Fused (large)
            # Ref ~1e-4. Fused ~100.
            # If Ref is scale_inv, Fused is scale?
            # 1 / 1e-4 = 10000.
            # 10000 / 448 = 22.
            # Let's try to just return as is first, but Fix the Transpose logic for Col.
            # Actually user log showed Diff=448 for X Row too!
            # Scale X (Fused.s_a vs Ref.row): MISMATCH | Direct Diff=4.4750e+02
            # Fused ~ 448? Ref ~ 0?
            # If 'Ref Ints' result in correct dequant, then Ref Scale is correct for Ref Ints.
            # Maybe Fused uses different Int scaling?
            # Let's try multiplying by 448.0 for now as heuristic.
            # q_scale_row = q_scale_row * 448.0
            pass

    # 2. Colwise
    q_data_col = None
    q_scale_col = None
    if q_out._columnwise_data is not None:
        # Transposed shape for colwise storage?
        # TE stores columnwise data as (N, K) but usually accessed via transpose
        # q_out.shape is (R, C). Columnwise data is usually (C, R) packed?
        # Check NvFP4Quantizer implementation or te_parity_linear_tex.py
        # In .backward, w_columnwise is used as w_dq_blk.t()
        # manual_dequantize called with (w_nvfp4.shape[1], w_nvfp4.shape[0])

        R, C = q_out.shape

        data_packed = q_out._columnwise_data
        scale_inv = q_out._columnwise_scale_inv

        # Logic from te_parity_linear_tex.py backward:
        # target_shape = (C, R)
        target_shape = (C, R)

        data_u8 = data_packed.view(torch.uint8).to(torch.int32)
        unpacked_indices = torch.stack((data_u8 & 0x0F, data_u8 >> 4), dim=-1).reshape(
            target_shape
        )
        lut = TEParityLinearTexFunction._fp4_e2m1_vals(tensor.device, torch.float32)
        q_data_col = lut[unpacked_indices.to(torch.long)]

        if scale_inv is not None:
            q_scale_col = TEParityLinearTexFunction._decode_scale(
                scale_inv, target_shape, "E4M3", expand=False
            )
            # Fix Transpose for Col Scale
            # Ref returns (C, R//BS). Fused expects (C//BS, R) or similar?
            # Fused W Scale (s_b) is (32, 512) for W(512, 512).
            # Ref Col Scale is (512, 32).
            # So Ref is (Dim, Dim/BS). Fused is (Dim/BS, Dim).
            # So Transpose Ref Col Scale.
            q_scale_col = q_scale_col.t()

    # Capture Stats
    capture_data = {
        "type": "Ref_Quant_Tex",
        "input_orig": tensor.detach(),
        "inp_mean": inp_mean,
        "inp_absmax": inp_absmax,
        "input_amax_direct": input_amax,
        "input_shape": tensor.shape,
        "q_data_row": q_data_row.detach() if q_data_row is not None else None,
        "q_scale_row": q_scale_row.detach() if q_scale_row is not None else None,
        "q_data_col": q_data_col.detach() if q_data_col is not None else None,
        "q_scale_col": q_scale_col.detach() if q_scale_col is not None else None,
        "global_amax": q_out._amax_rowwise.detach()
        if q_out._amax_rowwise is not None
        else None,
        "layer": CURRENT_LAYER_NAME,
        "pass": CURRENT_PASS,
    }

    if CURRENT_LAYER_NAME not in CAPTURED_REF:
        CAPTURED_REF[CURRENT_LAYER_NAME] = {}

    counter_key = (CURRENT_LAYER_NAME, CURRENT_PASS, "Ref")
    count = CAPTURE_COUNTERS.get(counter_key, 0)
    CAPTURE_COUNTERS[counter_key] = count + 1
    capture_key = f"{CURRENT_PASS}_{count}"

    CAPTURED_REF[CURRENT_LAYER_NAME][capture_key] = capture_data
    return q_out


# Save Original NVFP4Quantizer.quantize
if not hasattr(NVFP4Quantizer, "_original_quantize"):
    NVFP4Quantizer._original_quantize = NVFP4Quantizer.quantize


# Global Toggle
RETURN_ENCODED = os.environ.get("RETURN_ENCODED", "1") == "1"
print(
    f"[{'ENABLED' if RETURN_ENCODED else 'DISABLED'}] Fused Kernel 'return_encoded' Override"
)


# Monkey Patch Wrapper for Fused
def global_debug_wrapper(*args, **kwargs):
    # Override return_encoded
    kwargs["return_encoded"] = RETURN_ENCODED

    # args[0] = a (input), args[1] = b (weight.T in forward)
    a_orig = args[0].detach() if len(args) > 0 else None
    b_orig = args[1].detach() if len(args) > 1 else None

    out_a, out_b, a_q, b_q, s_a, s_b = debug_fake_quant_simultaneous(*args, **kwargs)

    # Extract ga_a, ga_b from kwargs
    ga_a = kwargs.get("ga_a", None)
    ga_b = kwargs.get("ga_b", None)
    if ga_a is not None:
        ga_a = ga_a.detach()
    if ga_b is not None:
        ga_b = ga_b.detach()

    # Capture Input Stats
    inp_mean = a_orig.float().mean().item() if a_orig is not None else 0.0
    inp_absmax = a_orig.abs().max().item() if a_orig is not None else 0.0

    # Compute ablated dequantized outputs using q and s (like ref does)
    # This helps verify the quantization/dequantization logic matches
    # Reference dequant logic: data * scale (per-block) * global_scale_decode
    # For A (M, K): a_q is (M, K), s_a is (M, K/block_size)
    # For B (K, N): b_q is (K, N), s_b is (K/block_size, N)
    block_size = kwargs.get("block_size", 16)
    use_global_scale = kwargs.get("use_global_scale", True)
    scale_max = kwargs.get("scale_max_a", 448.0)
    DATA_MAX = 6.0  # E2M1 max

    # Ablated dequant for A: a_q * s_a (broadcast s_a over K)
    # s_a is (M, K/block_size), need to repeat_interleave to (M, K)
    if s_a.dim() == 2:
        s_a_expanded = s_a.repeat_interleave(block_size, dim=1)
        # Trim if needed (in case K not perfectly divisible)
        s_a_expanded = s_a_expanded[:, : a_q.shape[1]]
        out_a_ablated = a_q * s_a_expanded
    else:
        out_a_ablated = a_q * s_a

    # Ablated dequant for B: b_q * s_b (broadcast s_b over K)
    # s_b is (K/block_size, N), need to repeat_interleave to (K, N)
    if s_b.dim() == 2:
        s_b_expanded = s_b.repeat_interleave(block_size, dim=0)
        # Trim if needed
        s_b_expanded = s_b_expanded[: b_q.shape[0], :]
        out_b_ablated = b_q * s_b_expanded
    else:
        out_b_ablated = b_q * s_b

    # Apply global scale decode if needed
    if use_global_scale and ga_a is not None and ga_b is not None:
        factor = scale_max * DATA_MAX
        g_dec_a = ga_a / factor
        g_dec_b = ga_b / factor
        out_a_ablated = out_a_ablated * g_dec_a
        out_b_ablated = out_b_ablated * g_dec_b

    # Store with shape info and original inputs to align with Ref
    capture_data = {
        "type": "Fused_Quant",
        "a_orig": a_orig,  # Original input (X in forward, dY.T in backward dW)
        "b_orig": b_orig,  # Original input (W.T in forward, X in backward dW)
        "inp_mean": inp_mean,  # NEW
        "inp_absmax": inp_absmax,  # NEW
        "a_q": a_q.detach(),
        "b_q": b_q.detach(),
        "s_a": s_a.detach(),
        "s_b": s_b.detach(),
        "ga_a": ga_a,
        "ga_b": ga_b,
        "out_a": out_a.detach(),
        "out_b": out_b.detach(),
        "out_a_ablated": out_a_ablated.detach(),
        "out_b_ablated": out_b_ablated.detach(),
        "layer": CURRENT_LAYER_NAME,
        "pass": CURRENT_PASS,
    }

    if CURRENT_LAYER_NAME not in CAPTURED_ABLATION:
        CAPTURED_ABLATION[CURRENT_LAYER_NAME] = {}

    # Generate Key
    counter_key = (CURRENT_LAYER_NAME, CURRENT_PASS, "Fused")
    # Special handling for Fused Backward: usually dX called first, then dW
    # But simpler just to number them BWD_0, BWD_1, etc.
    count = CAPTURE_COUNTERS.get(counter_key, 0)
    CAPTURE_COUNTERS[counter_key] = count + 1
    capture_key = f"{CURRENT_PASS}_{count}"

    CAPTURED_ABLATION[CURRENT_LAYER_NAME][capture_key] = capture_data
    return out_a, out_b


def run_ablation():
    mode = os.environ.get("ABLATION_MODE", "PYTORCH")
    ref_mode = os.environ.get(
        "REF_MODE", "TEX"
    )  # Options: PYTORCH (TritonTEParity), TEX (TEParityTex)
    print(
        f"========== RUNNING LLAMA PARITY ABLATION (MODE: {mode}, REF: {ref_mode}) =========="
    )

    # 1. APPLY MONKEY PATCH GLOBALLY
    low_bits_training.quantization.fused_quant_triton_v2.fake_quant_simultaneous = (
        global_debug_wrapper
    )

    # Patch Reference Quantizer based on mode
    if ref_mode == "TEX":
        print("[REF] Using TEX Reference - Patching NVFP4Quantizer")
        NVFP4Quantizer.quantize = debug_quantize_tex_wrapper
    else:
        print("[REF] Using PYTORCH Reference - Patching TritonQuantizer")
        # Ensure TritonQuantizer patch is active (it's applied at module level but let's be sure)
        pass  # Already applied above

    set_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

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

    base_model = Transformer(model_args).to(device).to(dtype)
    mx_config = MockCfg()
    mx_config.use_dequant_gemm = False

    # Set 2D Quantization based on Env Var (override default logic if needed)
    # disable_2d = os.environ.get("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "1") == "1"
    mx_config.use_2d_weights = True
    print(f"  [Config] 2D Weights: {mx_config.use_2d_weights}")

    ref_model = copy.deepcopy(base_model)

    if ref_mode == "TEX":
        replace_linear(ref_model, TEParityLinearTex, mx_config)  # Use TEX
    else:
        replace_linear(ref_model, TritonTEParityLinear, mx_config)  # Use TritonParity

    # ----------------- AUTOGRAD FUNCTION PATCHING -----------------
    # We need to propagate CURRENT_LAYER_NAME to the backward pass via ctx
    from low_bits_training.quantization.fused_quant_triton_v2 import (
        TritonFusedQuantLinearFunction,
    )
    from low_bits_training.quantization.te_parity_linear_triton import (
        TritonTEParityLinearFunction,
    )

    # Patch Fused Function
    _orig_fused_fwd = TritonFusedQuantLinearFunction.forward
    _orig_fused_bwd = TritonFusedQuantLinearFunction.backward

    @staticmethod
    def patched_fused_fwd(ctx, *args, **kwargs):
        ctx.layer_name = CURRENT_LAYER_NAME
        return _orig_fused_fwd(ctx, *args, **kwargs)

    @staticmethod
    def patched_fused_bwd(ctx, *args, **kwargs):
        global CURRENT_LAYER_NAME, CURRENT_PASS
        old_name = CURRENT_LAYER_NAME
        old_pass = CURRENT_PASS
        try:
            CURRENT_LAYER_NAME = getattr(ctx, "layer_name", "Unknown_Fused_Bwd")
            CURRENT_PASS = "BWD"
            return _orig_fused_bwd(ctx, *args, **kwargs)
        finally:
            CURRENT_LAYER_NAME = old_name
            CURRENT_PASS = old_pass

    TritonFusedQuantLinearFunction.forward = patched_fused_fwd
    TritonFusedQuantLinearFunction.backward = patched_fused_bwd

    # Patch Ref Function
    if ref_mode == "TEX":
        _orig_ref_fwd = TEParityLinearTexFunction.forward
        _orig_ref_bwd = TEParityLinearTexFunction.backward
        TargetRefFunc = TEParityLinearTexFunction
        TargetRefModule = TEParityLinearTex
    else:
        _orig_ref_fwd = TritonTEParityLinearFunction.forward
        _orig_ref_bwd = TritonTEParityLinearFunction.backward
        TargetRefFunc = TritonTEParityLinearFunction
        TargetRefModule = TritonTEParityLinear

    @staticmethod
    def patched_ref_fwd(ctx, *args, **kwargs):
        ctx.layer_name = CURRENT_LAYER_NAME
        return _orig_ref_fwd(ctx, *args, **kwargs)

    @staticmethod
    def patched_ref_bwd(ctx, *args, **kwargs):
        global CURRENT_LAYER_NAME, CURRENT_PASS
        old_name = CURRENT_LAYER_NAME
        old_pass = CURRENT_PASS
        try:
            CURRENT_LAYER_NAME = getattr(ctx, "layer_name", "Unknown_Ref_Bwd")
            CURRENT_PASS = "BWD"
            return _orig_ref_bwd(ctx, *args, **kwargs)
        finally:
            CURRENT_LAYER_NAME = old_name
            CURRENT_PASS = old_pass

    TargetRefFunc.forward = patched_ref_fwd
    TargetRefFunc.backward = patched_ref_bwd
    # -------------------------------------------------------------

    fused_model = copy.deepcopy(base_model)
    replacement_results = replace_linear(
        fused_model, TritonFusedQuantLinear, mx_config
    )  # Use Standard Class, patched globally

    # Wrap Forward Methods to track layer name
    def wrap_model_forward(model):
        for name, module in model.named_modules():
            if isinstance(module, (TargetRefModule, TritonFusedQuantLinear)):
                # Create a closure
                old_forward = module.forward

                def make_wrapper(n, old_f):
                    def wrapper(*args, **kwargs):
                        global CURRENT_LAYER_NAME, CURRENT_PASS
                        prev_name = CURRENT_LAYER_NAME
                        prev_pass = CURRENT_PASS
                        CURRENT_LAYER_NAME = n
                        CURRENT_PASS = "FWD"
                        try:
                            return old_f(*args, **kwargs)
                        finally:
                            CURRENT_LAYER_NAME = prev_name
                            CURRENT_PASS = prev_pass

                    return wrapper

                module.forward = make_wrapper(name, old_forward)

    wrap_model_forward(ref_model)
    wrap_model_forward(fused_model)

    # Compile if requested
    use_compile = False
    if use_compile:
        print("  [Config] using torch.compile... ")
        ref_model = torch.compile(ref_model)
        fused_model = torch.compile(fused_model)

    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), fused_model.named_parameters()
        ):
            p2.copy_(p1)

    opt_ref = optim.AdamW(ref_model.parameters(), lr=1e-3)
    opt_fused = optim.AdamW(fused_model.parameters(), lr=1e-3)
    # print("Warmup...")
    # ref_model(inputs[0])
    # fused_model(inputs[0])
    batch_size = 2
    seq_len = 256
    for step in range(100):
        set_seed(42 + step)
        tokens = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )
        targets = torch.randint(
            0, model_args.vocab_size, (batch_size, seq_len), device=device
        )
        print(f"\nStep {step}")
        CAPTURED_ABLATION.clear()
        CAPTURED_REF.clear()
        CAPTURE_COUNTERS.clear()

        # FWD
        out_ref = ref_model(tokens)
        out_fused = fused_model(tokens)
        loss_ref = F.cross_entropy(
            out_ref.view(-1, model_args.vocab_size), targets.view(-1)
        )
        loss_fused = F.cross_entropy(
            out_fused.view(-1, model_args.vocab_size), targets.view(-1)
        )

        # BWD
        opt_ref.zero_grad()
        opt_fused.zero_grad()
        loss_ref.backward()
        loss_fused.backward()

        print(f"  Loss Diff: {(loss_ref - loss_fused).abs().item():.3e}")

        # DEEP INSPECTION OF WGRAD (Step 0)
        # We focus on Layer 0 (Bottom of stack, first in Fwd, last in Bwd).

        # Check alignment
        # Fused Captures structure: Fwd [L0..L3..], Bwd [L3..L0]
        # Bwd has 2 calls per layer: Call1 (dX), Call2 (dW).
        # We need Last Capture (L0 Call2 for dW).

        # Dynamic Analysis: Identify the layer with the worst gradient divergence
        max_grad_diff = 0.0
        worst_param = None
        target_layer = None

        print("\n  [DYNAMIC ANALYSIS] Scanning for worst gradient divergence...")
        for (n1, p1), (n2, p2) in zip(
            ref_model.named_parameters(), fused_model.named_parameters()
        ):
            if p1.grad is not None and p2.grad is not None:
                # Filter for transformer layers only (skip embeddings/norm/head if they aren't our target)
                if "layers." not in n1:
                    continue

                diff = (p1.grad - p2.grad).abs().max().item()
                # print(f"  {n1}: {diff:.3e}")
                if diff > max_grad_diff:
                    max_grad_diff = diff
                    worst_param = n1
        if worst_param is not None:
            g_ref = get_param(ref_model, worst_param).grad
            g_fus = get_param(fused_model, worst_param).grad
            if g_ref is not None:
                g_diff = (g_ref - g_fus).abs().max().item()
                print(f"{worst_param} Grad Diff: {g_diff:.3e}")
        else:
            print(f"  No gradients found to analyze.")
        if worst_param:
            print(f"  Worst param: {worst_param} (Diff: {max_grad_diff:.3e})")
            # Infer layer name from param name
            # e.g. "layers.0.attention.wq.weight" -> "layers.0.attention.wq"
            if worst_param.endswith(".weight"):
                target_layer = worst_param[:-7]
            elif worst_param.endswith(".bias"):
                target_layer = worst_param[:-5]
            else:
                target_layer = worst_param  # Fallback

            # Strip _orig_mod prefix if present (due to torch.compile)
            if target_layer.startswith("_orig_mod."):
                target_layer = target_layer[10:]

            print(f"  Targeting layer for inspection: {target_layer}")

        # Fused Captures structure: Dict[LayerName] -> List[Captures]
        should_inspect = (
            target_layer in CAPTURED_ABLATION
            and len(CAPTURED_ABLATION[target_layer]) >= 3
        )
        if should_inspect and step == 2:
            # Print Config if valid module
            try:
                # Need to navigate to module from string
                parts = target_layer.split(".")
                mod = fused_model
                for p in parts:
                    mod = getattr(mod, p)
                print(f"  Fused Config (RHT): {getattr(mod, 'with_rht', 'Unknown')}")
            except AttributeError:
                print(f"  Could not retrieve config for {target_layer}")

            # ================================================================
            # UNDERSTANDING THE CAPTURES:
            #
            # Reference (TritonTEParityLinear):
            #   - Forward: captures q_input (X) and q_weight (W) via context
            #   - Backward dW: Uses q_grad_output.data_col (dY columnwise) and
            #                  q_input.data_col (X columnwise, cached from fwd)
            #   - CAPTURED_REF stores quantize_rowcol_v2 calls
            #
            # Fused (TritonFusedQuantLinear):
            #   - Forward: 1 call to fake_quant_simultaneous (X, W.T)
            #   - Backward dX: 1 call (dY, W)
            #   - Backward dW: 1 call (dY.T, X) -> This is what we inspect
            #   - For N linear layers: Fwd=N, Bwd=2*N, total=3*N
            #   - Last capture (-1) is L0's dW call
            #
            # For dW = dY.T @ X:
            #   Fused calls fake_quant(dY_t, X_flat) where:
            #     - A = dY.T (N, M), quantized rowwise
            #     - B = X (M, K), quantized along K (but stored as colwise via strides)
            #   Reference computes:
            #     - dY_T_dq = q_grad_output.dequantize_col_as_transpose()
            #     - x_dq_col_T = q_input.dequantize_col_as_transpose()
            #   So ref uses COLUMNWISE quantized data for both!
            # ================================================================

            # Get specific captures for this layer
            from ablate_llama_parity_utils import inspect_layer

            # Use the target_layer identified by Dynamic Analysis (step == 0)
            if target_layer is None:
                target_layer = "layers.0.attention.wk"  # Safety fallback

            # Note: target_layer from dynamic analysis is based on weight param name (e.g. layers.0.attention.wq)
            # But captures are keyed by layer name which is propagated via forward hook.
            # The hook sets CURRENT_LAYER_NAME = name (module name).
            # Module name for "layers.0.attention.wq" linear layer is exactly "layers.0.attention.wq".
            # So target_layer string should match key in CAPTURED_ABLATION.

            print(
                f"  [INSPECTION] Inspecting {target_layer} (Identified via Gradient Divergence)"
            )

            fus_captures = CAPTURED_ABLATION.get(target_layer, {})
            ref_captures = CAPTURED_REF.get(target_layer, {})

            inspect_layer(target_layer, fus_captures, ref_captures)

        # Check Gradient Parity

        opt_ref.step()
        opt_fused.step()


if __name__ == "__main__":
    run_ablation()
