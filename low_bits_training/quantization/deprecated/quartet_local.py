import torch
import qutlass

from scipy.linalg import hadamard
# Import from the submodule's package assuming it is in sys.path
from quartet2.quant import quant_fp4, quant_had_eden, quant_had_eden_dual, dequant_tp_had_eden
import nvtx
import contextlib

def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
        )


def nvtx_annotate(name: str, color: str = "green"):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    else:
        return nvtx.annotate(name, color=color)


def rerotate_hadamard(hadamard_matrix, generator=None):
    """Rerotate hadamard matrix with random signs. CUDA graph compatible."""
    # Use torch.rand which is CUDA graph compatible (not Generator-based)
    # Round to 0 or 1, then convert to -1/+1
    signs = torch.rand(hadamard_matrix.size(0), device=hadamard_matrix.device, dtype=hadamard_matrix.dtype)
    signs = (signs >= 0.5).to(hadamard_matrix.dtype) * 2 - 1
    return hadamard_matrix * signs[None, :]  # NOTE: rerotate along last dim, inner dim for TN GEMM


@torch.library.custom_op("clover::fp4_mm", mutates_args=(), tags=[torch._C.Tag.cudagraph_unsafe])
def _fp4_mm(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    # Ensure scales are viewed as float8 (binding requirement)
    if x_mx.dtype != torch.float8_e4m3fn:
        x_mx = x_mx.view(torch.float8_e4m3fn)
    if w_mx.dtype != torch.float8_e4m3fn:
        w_mx = w_mx.view(torch.float8_e4m3fn)

    return qutlass.matmul_nvf4_bf16_tn(
        x_fp4, w_fp4,
        x_mx, w_mx,
        alpha)


@_fp4_mm.register_fake
def _fp4_mm_fake(x_fp4: torch.Tensor, w_fp4: torch.Tensor, x_mx: torch.Tensor, w_mx: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.empty((x_fp4.shape[0], w_fp4.shape[0]), device=x_fp4.device, dtype=torch.bfloat16)



_to_blocked_idx_cache = {}

def to_blocked(input_matrix) -> torch.Tensor:
    """Rearrange micro-scales into blocked layout for FP4 GEMM.
    Uses a cached index permutation so each call is a single gather kernel."""
    rows, cols = input_matrix.shape
    key = (rows, cols, input_matrix.device)
    if key not in _to_blocked_idx_cache:
        n_row_blocks = rows // 128
        n_col_blocks = cols // 4
        idx = torch.arange(rows * cols, device=input_matrix.device)
        blocks = idx.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
        rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
        _to_blocked_idx_cache[key] = rearranged.flatten().contiguous()
    return input_matrix.flatten()[_to_blocked_idx_cache[key]]


# Removed @torch.compile
def _dq_fp4(x_e2m1: torch.Tensor, x_e4m3: torch.Tensor, alpha: float):
    device = x_e2m1.device

    x_e2m1_i32 = x_e2m1.view(dtype=torch.uint8).to(dtype=torch.int32)
    x_e2m1_unpacked = torch.stack(
        [x_e2m1_i32 & 0xF, (x_e2m1_i32 >> 4) & 0xF], dim=-1
    ).flatten(start_dim=-2)

    grid_dq = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    x_fp4_dq = grid_dq[x_e2m1_unpacked]

    scales_dq = x_e4m3.to(torch.float32)
    x_dq = (x_fp4_dq.unflatten(dim=-1, sizes=(-1, 16)) * scales_dq[..., None]).flatten(
        start_dim=-2
    ) * alpha
    return x_dq.to(torch.bfloat16)

# Removed @torch.compile
def abs_max(x):
    return x.abs().max().to(torch.float32)




class Quartet_II_fn(torch.autograd.Function):
    group_size = 16

    # Removed @torch.compile
    @staticmethod
    def forward(ctx, input, weight, had, four_over_six: bool, disable_backward_quant: bool = False, weight_amax: torch.Tensor = None, input_amax: torch.Tensor = None, scratch_amax: torch.Tensor = None):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.disable_backward_quant = disable_backward_quant
        ctx.four_over_six = four_over_six
        ctx.scratch_amax = scratch_amax
        assert input.dtype == torch.bfloat16

        forward_scale_override = 1.0

        flat_input = input.reshape(-1, input.shape[-1])
        
        with nvtx_annotate("Abs-max", color="red"):
            if input_amax is None:
                input_amax = abs_max(flat_input)
            if weight_amax is None:
                weight_amax = abs_max(weight)

        with nvtx_annotate("Quant", color="yellow"):
            # Ensure amax is float32
            if input_amax.dtype != torch.float32:
                 input_amax = input_amax.to(torch.float32)
            if weight_amax.dtype != torch.float32:
                 weight_amax = weight_amax.to(torch.float32)

            input_fp4 = quant_fp4(flat_input, amax=input_amax, scale_override=forward_scale_override, four_over_six=four_over_six)
            weight_fp4 = quant_fp4(weight, amax=weight_amax, scale_override=forward_scale_override, four_over_six=four_over_six)
        # TODO save quantized for requant kernels
        ctx.save_for_backward(input_fp4.fp4, input_fp4.micro_scales, input_fp4.tensor_scale,
                              weight_fp4.fp4, weight_fp4.micro_scales, weight_fp4.tensor_scale, had)
        with nvtx_annotate("Matmul", color="blue"):
            res = _fp4_mm(
                input_fp4.fp4, weight_fp4.fp4,
                to_blocked(input_fp4.micro_scales), to_blocked(weight_fp4.micro_scales),
                alpha=input_fp4.tensor_scale * weight_fp4.tensor_scale)
            
            # Ensure output is BF16
            if res.dtype != torch.bfloat16:
                res = res.to(torch.bfloat16)

        return res.reshape(ctx.batch, ctx.seq, ctx.out_dim)

    # Removed @torch.compile
    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        xfp4, xs, xm, wfp4, ws, wm, had = ctx.saved_tensors
        backward_scale_override = (17 / 16) * 0.93

        # Re-randomize the rotation (CUDA graph compatible - uses torch.rand)
        had = rerotate_hadamard(had)
        
        # FIXED: Ensure grad_output is BF16 for quant_had_eden
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)
        
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.disable_backward_quant:
            xr = _dq_fp4(xfp4, xs, xm)
            wr = _dq_fp4(wfp4, ws, wm)
            grad_input = flat_grad_output @ wr
            grad_weight = flat_grad_output.T @ xr
            return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None, None

        # Dual eden transform: read grad_output ONCE, produce both normal and transposed FP4
        with nvtx_annotate("Quant_Dual", color="yellow"):
            (e_ht_fp4, e_ht_ms, e_ht_ts), (et_ht_fp4, et_ht_ms, et_ht_ts) = quant_had_eden_dual(
                x=flat_grad_output, h=had, scale_override=backward_scale_override)

        # EW: grad_input = (grad_output @ H^T) @ (W^T @ H^T)^T
        with nvtx_annotate("Quant", color="yellow"):
            wt_ht_fp4, wt_ht_ms, wt_ht_ts = dequant_tp_had_eden(x=wfp4, x_group_scales=ws, x_tensor_scale=wm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_input = _fp4_mm(e_ht_fp4, wt_ht_fp4, to_blocked(e_ht_ms), to_blocked(wt_ht_ms), alpha=e_ht_ts*wt_ht_ts)
            if grad_input.dtype != torch.bfloat16:
                grad_input = grad_input.to(torch.bfloat16)

        # EtX: grad_weight = (grad_output^T @ H^T) @ (X^T @ H^T)^T
        with nvtx_annotate("Quant", color="yellow"):
            xt_ht_fp4, xt_ht_ms, xt_ht_ts = dequant_tp_had_eden(x=xfp4, x_group_scales=xs, x_tensor_scale=xm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_weight = _fp4_mm(et_ht_fp4, xt_ht_fp4, to_blocked(et_ht_ms), to_blocked(xt_ht_ms), alpha=et_ht_ts*xt_ht_ts)
            if grad_weight.dtype != torch.bfloat16:
                grad_weight = grad_weight.to(torch.bfloat16)
        return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None


# CUDA graph compatible - removed @torch.compiler.disable
class QuartetLinearLocal(torch.nn.Linear):
    """FP4 Linear layer using Quartet-II quantization with manual prefetch support."""
    
    def __init__(self, *args, four_over_six=True, dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, dtype=dtype, **kwargs)
        assert dtype == torch.bfloat16
        self.four_over_six = four_over_six
        self.weight_abs_max = None
        self.register_buffer("had", get_hadamard_matrix(128, self.weight.dtype, self.weight.device))
        # FIXED: scratch_amax is uint32 scalar to match quant.py requirements
        self.register_buffer("scratch_amax", torch.zeros((), dtype=torch.uint32, device=self.weight.device))
        
        # Manual prefetch support for AllGather overlap
        self._prefetch_handle = None
        self._next_layer = None  # Reference to next layer for prefetch chaining
        
    def set_next_layer(self, next_layer: 'QuartetLinearLocal'):
        """Set reference to next layer for prefetch chaining."""
        self._next_layer = next_layer
        
    def prefetch_weights(self):
        """Start async AllGather for this layer's weights.
        
        Call this during the previous layer's compute to overlap communication.
        Uses FSDP2's unshard() API if available.
        """
        # Try FSDP2 API first (fully_shard wraps modules with FSDPModule)
        from torch.distributed._composable.fsdp import FSDPModule
        
        # Check if this module is wrapped by FSDP
        if hasattr(self, '_fsdp_state') or isinstance(self, FSDPModule):
            try:
                # FSDP2 exposes unshard() for manual control
                self._prefetch_handle = self.unshard(async_op=True)
            except (AttributeError, TypeError):
                # Fallback: FSDP may not expose unshard directly
                pass
    
    def wait_prefetch(self):
        """Wait for prefetched weights to be ready."""
        if self._prefetch_handle is not None:
            try:
                self._prefetch_handle.wait()
            except AttributeError:
                pass
            self._prefetch_handle = None
            
    def forward(self, x, disable_backward_quant=False, input_abs_max=None):
        # Wait for any pending prefetch before using weights
        self.wait_prefetch()
        
        # Trigger prefetch for next layer (overlaps with our compute)
        if self._next_layer is not None:
            self._next_layer.prefetch_weights()
            
        # Auto-cast input to BF16 if needed
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        if not x.is_contiguous():
            x = x.contiguous()
            
        out = Quartet_II_fn.apply(x, self.weight, self.had, self.four_over_six, disable_backward_quant, self.weight_abs_max, input_abs_max, self.scratch_amax)
        
        # Ensure output is BF16 (just in case Quartet returns FP32)
        if out.dtype != torch.bfloat16:
             out = out.to(torch.bfloat16)

        if self.bias is not None:
            out = out + self.bias
            
        return out


def setup_quartet_prefetch_chain(model):
    """Set up prefetch chaining between Quartet layers in a model.
    
    Call this after FSDP wrapping to enable AllGather overlap.
    """
    quartet_layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuartetLinearLocal):
            quartet_layers.append((name, module))
    
    # Chain layers: each layer prefetches the next
    for i in range(len(quartet_layers) - 1):
        current_name, current_layer = quartet_layers[i]
        next_name, next_layer = quartet_layers[i + 1]
        current_layer.set_next_layer(next_layer)
    
    return len(quartet_layers)


class Quartet_II_PreQuant_fn(torch.autograd.Function):
    """Autograd function for pre-quantized weights (FP4)."""
    
    @staticmethod
    def forward(ctx, input, weight_fp4, weight_scales, weight_global_scale, had, four_over_six: bool, disable_backward_quant: bool = False, input_amax: torch.Tensor = None, scratch_amax: torch.Tensor = None):
        ctx.batch = input.shape[0]
        ctx.seq = input.shape[1]
        # Inferred dims from packed weight
        # weight_fp4 is (out, in // 2)
        ctx.out_dim = weight_fp4.shape[0]
        ctx.in_dim = weight_fp4.shape[1] * 2
        
        ctx.disable_backward_quant = disable_backward_quant
        ctx.four_over_six = four_over_six
        ctx.scratch_amax = scratch_amax
        assert input.dtype == torch.bfloat16

        forward_scale_override = 1.0

        flat_input = input.reshape(-1, input.shape[-1])
        
        with nvtx_annotate("Abs-max", color="red"):
            if input_amax is None:
                input_amax = abs_max(flat_input)

        with nvtx_annotate("Quant", color="yellow"):
            # Ensure amax is float32
            if input_amax.dtype != torch.float32:
                 input_amax = input_amax.to(torch.float32)

            input_fp4 = quant_fp4(flat_input, amax=input_amax, scale_override=forward_scale_override, four_over_six=four_over_six)
            
            # Weight is already quantized!
            
        # Save for backward
        # Reconstruct NVFP4Quant tuple for clarity if needed, but we save tensors
        ctx.save_for_backward(input_fp4.fp4, input_fp4.micro_scales, input_fp4.tensor_scale,
                              weight_fp4, weight_scales, weight_global_scale, had)

        with nvtx_annotate("Matmul", color="blue"):
            res = _fp4_mm(
                input_fp4.fp4, weight_fp4,
                to_blocked(input_fp4.micro_scales), to_blocked(weight_scales),
                alpha=input_fp4.tensor_scale * weight_global_scale)
            
            # Ensure output is BF16
            if res.dtype != torch.bfloat16:
                res = res.to(torch.bfloat16)

        return res.reshape(ctx.batch, ctx.seq, ctx.out_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx
        xfp4, xs, xm, wfp4, ws, wm, had = ctx.saved_tensors
        backward_scale_override = (17 / 16) * 0.93

        # Re-randomize the rotation
        had = rerotate_hadamard(had)
        
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)
        
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
            
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        if ctx.disable_backward_quant:
            wr = _dq_fp4(wfp4, ws, wm)
            grad_input = flat_grad_output @ wr
            xr = _dq_fp4(xfp4, xs, xm)
            grad_weight = flat_grad_output.T @ xr
            return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None, None, None

        # Dual eden transform: read grad_output ONCE, produce both normal and transposed FP4
        with nvtx_annotate("Quant_Dual", color="yellow"):
            (e_ht_fp4, e_ht_ms, e_ht_ts), (et_ht_fp4, et_ht_ms, et_ht_ts) = quant_had_eden_dual(
                x=flat_grad_output, h=had, scale_override=backward_scale_override)

        # EW: grad_input
        with nvtx_annotate("Quant", color="yellow"):
            wt_ht_fp4, wt_ht_ms, wt_ht_ts = dequant_tp_had_eden(x=wfp4, x_group_scales=ws, x_tensor_scale=wm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_input = _fp4_mm(e_ht_fp4, wt_ht_fp4, to_blocked(e_ht_ms), to_blocked(wt_ht_ms), alpha=e_ht_ts*wt_ht_ts)
            if grad_input.dtype != torch.bfloat16:
                grad_input = grad_input.to(torch.bfloat16)

        # EtX: grad_weight
        with nvtx_annotate("Quant", color="yellow"):
            xt_ht_fp4, xt_ht_ms, xt_ht_ts = dequant_tp_had_eden(x=xfp4, x_group_scales=xs, x_tensor_scale=xm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
        with nvtx_annotate("Matmul", color="blue"):
            grad_weight = _fp4_mm(et_ht_fp4, xt_ht_fp4, to_blocked(et_ht_ms), to_blocked(xt_ht_ms), alpha=et_ht_ts*xt_ht_ts)
            if grad_weight.dtype != torch.bfloat16:
                grad_weight = grad_weight.to(torch.bfloat16)
        return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None, None

class QuartetLinearFP4(QuartetLinearLocal):
    """FP4 Linear layer that quantizes BF16 weights on-the-fly in forward().
    
    This is the TE-like approach: FSDP manages a single BF16 weight parameter,
    and FP4 quantization happens after AllGather during the forward pass.
    This ensures identical AllGather volume and behavior as TE.
    
    Features:
      - Delayed input amax scaling: uses amax from previous forward for current
        quantization, eliminating the abs_max kernel from the critical path.
      - Weight quant caching: caches quantized weight between forward calls
        within the same training step.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, four_over_six=True, use_bf16_backward=False):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, four_over_six=four_over_six)
        self.use_bf16_backward = use_bf16_backward
        # Delayed input amax: initialized to 0 (meaning "not yet computed")
        self._delayed_input_amax = None
        # Weight quant cache
        self._weight_quant_cache = None
        self._weight_data_ptr = None  # Track weight identity for invalidation

    def _get_weight_quant(self, weight):
        """Get (possibly cached) quantized weight."""
        # Invalidate cache if weight tensor changed (after AllGather or optimizer)
        ptr = weight.data_ptr()
        if self._weight_quant_cache is not None and self._weight_data_ptr == ptr:
            return self._weight_quant_cache
        with torch.no_grad():
            amax = abs_max(weight)
            res = quant_fp4(weight, amax=amax, scale_override=1.0, four_over_six=self.four_over_six)
        self._weight_quant_cache = (res.fp4, res.micro_scales, res.tensor_scale)
        self._weight_data_ptr = ptr
        return self._weight_quant_cache

    def forward(self, x, disable_backward_quant=None, input_abs_max=None):
        if disable_backward_quant is None:
            disable_backward_quant = self.use_bf16_backward
        self.wait_prefetch()
        
        if self._next_layer is not None:
            self._next_layer.prefetch_weights()
            
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()

        # Cached weight quantization
        weight_fp4, weight_scales, weight_global_scale = self._get_weight_quant(self.weight)

        # Delayed input amax scaling: use previous forward's value if available
        if input_abs_max is None and self._delayed_input_amax is not None:
            input_abs_max = self._delayed_input_amax

        out = Quartet_II_PreQuant_fn.apply(
            x, 
            weight_fp4, 
            weight_scales, 
            weight_global_scale, 
            self.had, 
            self.four_over_six, 
            disable_backward_quant, 
            input_abs_max, 
            self.scratch_amax
        )

        # Update delayed amax for next forward (runs after GEMM is launched)
        with torch.no_grad():
            flat = x.reshape(-1, x.shape[-1])
            self._delayed_input_amax = abs_max(flat)
        
        # Ensure output is BF16
        if out.dtype != torch.bfloat16:
             out = out.to(torch.bfloat16)

        if self.bias is not None:
            out = out + self.bias

        return out


class FusedQuartetLinearFP4(QuartetLinearFP4):
    """FP4 Linear layer that absorbs the preceding RMSNorm into its forward pass.
    
    Instead of separate RMSNorm → abs_max → quant_fp4, this layer computes
    RMSNorm internally before quantization, reducing kernel launch overhead.
    
    The converter replaces the model's RMSNorm with nn.Identity and moves
    the norm weights into this layer.
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 four_over_six=True, use_bf16_backward=False, norm_eps=1e-5):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype,
                         four_over_six=four_over_six, use_bf16_backward=use_bf16_backward)
        self.norm_eps = norm_eps
        # RMSNorm weight absorbed from the preceding norm layer
        self.norm_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))

    def forward(self, x, disable_backward_quant=None, input_abs_max=None):
        """Forward with fused RMSNorm: input x has NOT been normalized yet."""
        if disable_backward_quant is None:
            disable_backward_quant = self.use_bf16_backward
        self.wait_prefetch()
        
        if self._next_layer is not None:
            self._next_layer.prefetch_weights()
            
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()

        # Apply RMSNorm inline (fused into this layer)
        x_norm = torch.nn.functional.rms_norm(x, (x.shape[-1],), self.norm_weight, eps=self.norm_eps)

        # Cached weight quantization
        weight_fp4, weight_scales, weight_global_scale = self._get_weight_quant(self.weight)

        # Delayed input amax scaling
        if input_abs_max is None and self._delayed_input_amax is not None:
            input_abs_max = self._delayed_input_amax

        out = Quartet_II_PreQuant_fn.apply(
            x_norm,
            weight_fp4, 
            weight_scales, 
            weight_global_scale, 
            self.had, 
            self.four_over_six, 
            disable_backward_quant, 
            input_abs_max, 
            self.scratch_amax
        )

        # Update delayed amax for next forward
        with torch.no_grad():
            flat = x_norm.reshape(-1, x_norm.shape[-1])
            self._delayed_input_amax = abs_max(flat)
        
        if out.dtype != torch.bfloat16:
             out = out.to(torch.bfloat16)

        if self.bias is not None:
            out = out + self.bias

        return out

