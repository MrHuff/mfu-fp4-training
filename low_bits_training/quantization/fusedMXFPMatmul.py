from low_bits_training.quantization.dimensionQuantisationClass import *
from low_bits_training.quantization.MXFPconfig import MXLinearDimConfig
from low_bits_training.quantization.mx_ops_dim import *
from low_bits_training.quantization.mxfp import _is_linear, replace_with_custom_fn_if_matches_filter

#TODO MXtensors in question. Need original shape for both tensors. Assume full tensors A and B come in and you do matmul - what gradient should get passed to both tensors, assuming the whole MXFP exercise?!
# Input levers: scaling type, block_size, hard or soft absmax 

# back prop levers: quantisation gradient type (ideally fast!), absmax approximation type, numerics_scale. 

#Big question: how to ensure correct gradient when block-scaling to arbitrary shapes?
#Assumptions:
# 1. a block is at most 1 or 2 dimensional. We can start easy in the simualted case to 1d blocks (defined by a number), but we should generalise
    # if 2d block, we have to specify a shape type, or even better - split the matrix along a dimension with a divisble number or even two dimensions!
# 2. It's matmul so we assume what comes are matrices, we idgaf about anything before that.
# 3. I really think you just concat/reshape the normalisation for each block and multiply with the big gradient!!!!
# 4. You can think about it this way - basically a block itself will yield a backwards gradient that you can backpropagate!
# 5. For the hard gradient case, think about how amax style gradient distribution is implemented, note that any max function only applied elementwise


#Assumption A = z x m , B = m x y
# dC = z x y  dC^T  = y x z


def mx_tensor_nan_check(mx_tensor: DimensionMXTensor, tensor_name: str):
    if torch.isnan(mx_tensor._data).any().item():
        print(f'{tensor_name} quantisation tensor has nans')

    if torch.isnan(mx_tensor._scale_fp).any().item():
        print(f'{tensor_name} scale_fp tensor has nans')

    if torch.isnan(mx_tensor._scale_lp).any().item():
        print(f'{tensor_name} scale_lp tensor has nans')

def e8m0_to_float(exponent_8bit):
    if exponent_8bit == 0:
        return 0.0  # No subnormals!
    elif exponent_8bit == 255:
        return np.nan  # Invalid (could also assert/error)
    else:
        return 2.0 ** (exponent_8bit - 127)
def e5m3_unsigned_to_float(exponent_5bit: int, mantissa_3bit: int) -> float:
    """
    Converts an E5M3 unsigned floating-point number to a standard Python float.
    This format has no sign bit.

    Args:
        exponent_5bit: The 5-bit exponent (0-31).
        mantissa_3bit: The 3-bit mantissa (0-7).

    Returns:
        The float representation.
    """
    # Bias for a 5-bit exponent is 2^(5-1) - 1 = 15.
    bias = 15
    
    # Special values: exponent is all 1s (31)
    if exponent_5bit == 31:
        if mantissa_3bit == 0:
            return float('inf')  # Infinity
        else:
            return float('nan')   # Not a Number

    # Subnormal numbers: exponent is 0
    if exponent_5bit == 0:
        # Value = 0.mantissa * 2^(1 - bias)
        # The mantissa is divided by 2^3 = 8
        return (mantissa_3bit / 8.0) * (2.0 ** (1 - bias)) # 2**-14
    
    # Normal numbers
    else:
        # Value = 1.mantissa * 2^(exponent - bias)
        # The mantissa is divided by 2^3 = 8
        return (1.0 + mantissa_3bit / 8.0) * (2.0 ** (exponent_5bit - bias))
def e5m2_to_float(sign_bit: int, exponent_5bit: int, mantissa_2bit: int) -> float:
    """
    Converts a standard E5M2 floating-point number to a standard Python float.

    Args:
        sign_bit: The 1-bit sign (0 for positive, 1 for negative).
        exponent_5bit: The 5-bit exponent (0-31).
        mantissa_2bit: The 2-bit mantissa (0-3).

    Returns:
        The float representation.
    """
    # Bias for a 5-bit exponent is 2^(5-1) - 1 = 15.
    bias = 15
    
    # Determine the sign
    sign = -1.0 if sign_bit == 1 else 1.0
    
    # Special values: exponent is all 1s (31)
    if exponent_5bit == 31:
        if mantissa_2bit == 0:
            return sign * float('inf')  # Infinity (can be positive or negative)
        else:
            return float('nan')        # Not a Number (usually no sign)

    # Subnormal numbers: exponent is 0
    if exponent_5bit == 0:
        # Value = (-1)^sign * 0.mantissa * 2^(1 - bias)
        # The mantissa is divided by 2^2 = 4
        return sign * (mantissa_2bit / 4.0) * (2.0 ** (1 - bias)) # 2**-14
    
    # Normal numbers
    else:
        # Value = (-1)^sign * 1.mantissa * 2^(exponent - bias)
        # The mantissa is divided by 2^2 = 4
        return sign * (1.0 + mantissa_2bit / 4.0) * (2.0 ** (exponent_5bit - bias))

def e4m3_to_float(exponent_4bit, mantissa_3bit):
    # E4M3 format (4-bit exponent, 3-bit mantissa)
    if exponent_4bit == 0:
        # Subnormal number (optional - depends on implementation)
        return (mantissa_3bit / 8.0) * 2.0 ** (-6)  # Minimum exponent -6
    else:
        # Normal number: 1.mantissa * 2^exponent
        return (1.0 + mantissa_3bit / 8.0) * 2.0 ** (exponent_4bit - 7)  # Bias = 7

def e8m3_to_float(exponent_8bit, mantissa_3bit):
    """
    Converts a theoretical E8M3 (8-bit exponent, 3-bit mantissa) to a float.
    """
    # E8M3 format (8-bit exponent, 3-bit mantissa)
    # Bias = 2^(8-1) - 1 = 127
    
    if exponent_8bit == 0:
        # Subnormal number: 0.mantissa * 2^(1 - bias)
        # Smallest exponent for normal numbers is 1 - 127 = -126
        return (mantissa_3bit / 8.0) * 2.0**(-126)
        
    elif exponent_8bit == 255:
        # Represents infinity or NaN. For simplicity, we'll return NaN.
        return np.nan
        
    else:
        # Normal number: 1.mantissa * 2^(exponent - bias)
        return (1.0 + mantissa_3bit / 8.0) * 2.0**(exponent_8bit - 127)
def e8m7_to_float(exponent_8bit, mantissa_7bit):
        """
        Converts a BFLOAT16 (E8M7) representation to a float.
        """
        # E8M7 format (8-bit exponent, 7-bit mantissa)
        # Bias = 2^(8-1) - 1 = 127
        
        if exponent_8bit == 255:
            # Represents infinity or NaN. For simplicity, we'll return NaN.
            return np.nan
        elif exponent_8bit == 0:
            # Subnormal number: 0.mantissa * 2^(1 - bias)
            # Divisor for 7-bit mantissa is 2^7 = 128
            return (mantissa_7bit / 128.0) * 2.0**(-126)
        else:
            # Normal number: 1.mantissa * 2^(exponent - bias)
            return (1.0 + mantissa_7bit / 128.0) * 2.0**(exponent_8bit - 127)
def getScaleGrid(config):
    # --- Start of updated code ---
    if config.scale_type == 'E8M0':
        valid_exponents = range(1, 255)
        e8m0_values = [e8m0_to_float(e) for e in valid_exponents]
        return np.array(e8m0_values[:20]), np.linspace(0, e8m0_values[20], 40)
        
    elif config.scale_type == 'E4M3':
        e4m3_values = []
        for exp in range(16):
            for mant in range(8):
                e4m3_values.append(e4m3_to_float(exp, mant))
        return np.array(e4m3_values[:20]), np.linspace(0, e4m3_values[20], 40)
    elif config.scale_type == 'E5M3':
        e4m3_values = []
        for exp in range(16):
            for mant in range(8):
                e4m3_values.append(e4m3_to_float(exp, mant))
        return np.array(e4m3_values[:20]), np.linspace(0, e4m3_values[20], 40)
    elif config.scale_type == 'E5M2':
        e4m3_values = []
        for exp in range(16):
            for mant in range(8):
                e4m3_values.append(e4m3_to_float(exp, mant))
        return np.array(e4m3_values[:20]), np.linspace(0, e4m3_values[20], 40)
    elif config.scale_type == 'E8M3':
        # Generate all possible E8M3 values
        e8m3_values = []
        for exp in range(256):  # 8-bit exponent (0-255)
            for mant in range(8):  # 3-bit mantissa (0-7)
                e8m3_values.append(e8m3_to_float(exp, mant))
        # Note: Depending on usage, you may want to filter out NaNs
        # e.g., e8m3_values = [v for v in e8m3_values if not np.isnan(v)]
        return np.array(e8m3_values[:20]), np.linspace(0, e8m3_values[20], 40)
    elif config.scale_type == 'E8M7':
        # Generate all possible E8M7 (BFLOAT16) values
        e8m7_values = []
        for exp in range(256):  # 8-bit exponent (0-255)
            for mant in range(128):  # 7-bit mantissa (0-127)
                e8m7_values.append(e8m7_to_float(exp, mant))
        return np.array(e8m7_values[:20]), np.linspace(0, e8m7_values[20], 40)
    elif config.scale_type == 'Ideal':
        return np.array([1.0]), np.array([1.0])
        
    else:
        raise ValueError(f"Unsupported scale type: {config.scale_type}")
class STEscaleGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
                grad: torch.Tensor ,
                lp_tensor: DimensionMXTensor,
                q_prime: torch.nn.Module,
                X:torch.Tensor
                ):
        return grad
    

class TensorScalingGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
                grad: torch.Tensor ,
                lp_tensor: DimensionMXTensor,
                X: torch.Tensor,
                grad_type: str,
                ):
        
        if grad_type =='absmax':
            return  grad  + lp_tensor._global_abs_mask.reshape(grad.shape) *  ( lp_tensor.to_dtype(X.dtype) - X/lp_tensor._g * grad)
        elif grad_type =='ignore':
            return grad  
        else :
            return grad + ( lp_tensor.to_dtype(X.dtype) - X/lp_tensor._g * grad)
            

class AbsMaxGradFPScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
                grad: torch.Tensor ,
                lp_tensor: DimensionMXTensor,
                q_prime: torch.nn.Module,
                X: torch.Tensor,
                ):
        reshaped_grad, grad_orig_shape = blockify(grad,lp_tensor._block_size)
        g_reshaped, _ = blockify(lp_tensor._data,lp_tensor._block_size)

        scaled_reshaped_grad = reshaped_grad * lp_tensor._scale_fp/lp_tensor._scale_lp  * lp_tensor._max_abs_mask
        scaled_reshaped_grad_2 = q_prime(lp_tensor._scale_fp,1.0)/lp_tensor._scale_lp**2   *  torch.logical_not(lp_tensor._max_abs_mask) * g_reshaped * (lp_tensor._scale_fp/lp_tensor._max_abs * torch.sign(g_reshaped))

        return (scaled_reshaped_grad + scaled_reshaped_grad_2).reshape(grad_orig_shape)

        #TODO, figure smooth quantisation for scale and additional term!
class SoftMaxGradFPScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
                grad: torch.Tensor ,
                lp_tensor: DimensionMXTensor,
                q_prime: torch.nn.Module,
                X: torch.Tensor,
                ):
        reshaped_grad, grad_orig_shape = blockify(grad, p_tensor._block_size)
        g_reshaped, _ = blockify(lp_tensor._data, lp_tensor._block_size)
        X_reshaped, _ = blockify(X, lp_tensor._block_size)
        scaled_reshaped_grad = reshaped_grad * lp_tensor._scale_fp/lp_tensor._scale_lp 
        dn = -lp_tensor._scale_lp/lp_tensor._max_abs  * lp_tensor._sm * torch.sign(X_reshaped)
        scaled_reshaped_grad_2 =  X_reshaped*reshaped_grad/lp_tensor._scale_lp - g_reshaped * q_prime(lp_tensor._scale_fp,1.0)/lp_tensor._scale_lp**2 
        #propagate softmax somehow, have to save the softmax annoyingly     
        
        return (scaled_reshaped_grad+dn * scaled_reshaped_grad_2).reshape(grad_orig_shape)
            #return (scaled_reshaped_grad+scaled_reshaped_grad_2).reshape(grad_orig_shape)

class AbsMaxGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
                grad: torch.Tensor ,
                lp_tensor: DimensionMXTensor,
                q_prime: torch.nn.Module,
                X: torch.Tensor
                ):
        reshaped_grad, grad_orig_shape = blockify(grad, lp_tensor._block_size)
        g_reshaped, _ = blockify(lp_tensor._data,lp_tensor._block_size)

        scaled_reshaped_grad_2 =  q_prime(lp_tensor._scale_fp,1.0) /lp_tensor._scale_lp * torch.logical_not(lp_tensor._max_abs_mask) * (lp_tensor._scale_fp /lp_tensor._max_abs * torch.sign(g_reshaped)/lp_tensor._scale_lp * g_reshaped  - lp_tensor._scale_fp  * reshaped_grad )

        return (reshaped_grad + scaled_reshaped_grad_2).reshape(grad_orig_shape)

class SoftMaxGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,
                grad: torch.Tensor ,
                lp_tensor: DimensionMXTensor,
                q_prime: torch.nn.Module,
                X: torch.Tensor
                ):
        X_reshaped, _ = blockify(X,lp_tensor._block_size)
        g_reshaped, _ = blockify(lp_tensor._data,lp_tensor._block_size)
        reshaped_grad, grad_orig_shape = blockify(grad,lp_tensor._block_size)
        dn = -lp_tensor._scale_lp/lp_tensor._max_abs  * lp_tensor._sm * torch.sign(X_reshaped)
        scaled_reshaped_grad_2 =  (X_reshaped*reshaped_grad - g_reshaped / lp_tensor._scale_lp) * q_prime(lp_tensor._scale_fp,1.0)/lp_tensor._scale_lp
        #propagate softmax somehow, have to save the softmax annoyingly     
        
        return (reshaped_grad+dn * scaled_reshaped_grad_2).reshape(grad_orig_shape)
            #return (scaled_reshaped_grad+scaled_reshaped_grad_2).reshape(grad_orig_shape)

class STEquantisationGrad(torch.nn.Module):
    def __init__(selfe):
        super().__init__()

    def forward(self, data_hp_scaled ,g):
        return g

class BaselineQuantisationGrad(torch.nn.Module):
    def __init__(self,quant_values, k, dtype,lb,ub):
        super().__init__()
        self.k = k 
        self.dtype = dtype
        self.quant = EXMYQuantization(quant_values=quant_values , dtype=dtype)
        self.lb = lb
        self.ub = ub

    def forward(self, data_hp_scaled, g):
        max_mask = self.quant.max > data_hp_scaled
        # Find the index where x falls into the interval
        quant = self.quant.to(data_hp_scaled.device)
        idx = torch.bucketize(data_hp_scaled, quant.quant_values) - 1
        idx = torch.clamp(idx, 0, len(quant.deltas) - 1)

        # Extract corresponding values
        mid = quant.deltas[idx] / 2
        x_diff = (data_hp_scaled - quant.centers[idx])
        abs_x_diff = torch.abs(x_diff)
        nonzero_mask = abs_x_diff != 0
        grad_output = torch.zeros_like(x_diff)
        grad_output[nonzero_mask] = (
            mid[nonzero_mask] * (1 / self.k) * abs_x_diff[nonzero_mask] ** (1 / self.k - 1)
        )
        return (grad_output * torch.logical_not(max_mask) + max_mask) * g
    

class SigmoidQuantisationGrad(torch.nn.Module):
    def __init__(self,quant_values, temperature, dtype, lb=0.3,ub=1.5):
        super().__init__()
        self.temperature= temperature
        self.dtype = dtype
        self.quant = EXMYQuantization(quant_values,dtype)
        self.lb = lb
        self.ub = ub

    def forward(self, data_hp_scaled, g):
        max_mask = self.quant.max > data_hp_scaled
        # Find the index where x falls into the interval
        quant = self.quant.to(data_hp_scaled.device)

        idx = torch.bucketize(data_hp_scaled, quant.quant_values) - 1
        idx = torch.clamp(idx, 0, len(quant.deltas) - 1)

        # Extract corresponding values
        x_diff = (data_hp_scaled - quant.centers[idx]) * quant.sigmoidDeltas[idx]
        
        sigmoid_values = torch.sigmoid(x_diff / self.temperature)
        grad = ((sigmoid_values * (1 - sigmoid_values)) / self.temperature *12).clip(self.lb, self.ub)
        return (grad * torch.logical_not(max_mask) + max_mask) * g
    

class LinearSplineQuantisationGrad(torch.nn.Module):
    def __init__(self,quant_values,quant_func,x_test, dtype,lb,ub):
        super().__init__()
        self.dtype = dtype
        self.spline = SplineGradModule(quant_func=quant_func,x_test=x_test, dtype=dtype, K=1)
        self.quant = EXMYQuantization(quant_values,dtype)
        self.lb = lb
        self.ub = ub

    def forward(self, data_hp_scaled , g):
        max_mask = self.quant.max > data_hp_scaled
        # --- Existing setup code ---
        spline = self.spline.to(data_hp_scaled.device)
        indices = torch.searchsorted(spline.knots, data_hp_scaled, right=True) - 1
        indices = torch.clamp(indices, 0, len(spline.knots) - 2)
        offset = data_hp_scaled - spline.knots[indices]
        c_ind = spline.coeffs[:, indices]

        # --- START of the corrected logic ---

        # 1. Compute the correct coefficients for the derivative polynomial.
        # For a polynomial of degree `m`, the derivative's coefficients are derived
        # by multiplying the original coefficients (C_0, C_1, ...) by their corresponding
        # powers (m, m-1, ...).
        num_coeffs = c_ind.shape[0]
        degree = num_coeffs - 1

        if degree < 1:
            # If the spline is constant (degree 0), the derivative is zero.
            poly_derivative = torch.zeros_like(data_hp_scaled)
        else:
            # Create multipliers for the coefficients, i.e., [m, m-1, ..., 1]
            # We reshape it to work with coefficient tensors of any dimension.
            power_multipliers = torch.arange(degree, 0, -1, device=c_ind.device, dtype=c_ind.dtype)
            power_multipliers = power_multipliers.view(-1, *[1] * data_hp_scaled.dim())
            
            # The coefficients for the derivative polynomial.
            # We take all but the last coefficient of the original polynomial.
            deriv_coeffs = c_ind[:-1] * power_multipliers

            # 2. Evaluate the derivative polynomial using a fast, vectorized method.
            # This replaces the slow Python for-loop and works on tensors of any shape.
            
            # Create a matrix of the powers of the offset `h`: [h^(m-1), h^(m-2), ..., h^0]
            deriv_degree = degree - 1
            
            # Prepare exponent tensor, reshaping for broadcasting with the input tensor shape
            exponents = torch.arange(deriv_degree, -1, -1, device=c_ind.device, dtype=c_ind.dtype)
            exponents = exponents.view(-1, *[1] * data_hp_scaled.dim())
            
            # Broadcast `offset` to compute all powers at once. `offset` has shape (*),
            # `offset.unsqueeze(0)` has shape (1, *), `offset_powers` has shape (deriv_degree+1, *)
            offset_powers = offset.unsqueeze(0) ** exponents
            
            # Multiply coefficients by their corresponding offset powers and sum them up.
            # This is the vectorized equivalent of Horner's method.
            poly_derivative = torch.sum(deriv_coeffs * offset_powers, dim=0)

    # --- END of the corrected logic ---

        return (poly_derivative.clip(self.lb,self.ub) * torch.logical_not(max_mask) + max_mask) * g




class FusedMXFPMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                A: torch.Tensor,
                B: torch.Tensor, #B=W.T, Y = A@B
                config: MXLinearDimConfig,
                quantgrad: torch.nn.Module,
                scalegrad: torch.nn.Module,
                tensorscalegrad: torch.nn.Module,
                q_prime: torch.nn.Module,
                scalingModule: torch.nn.Module,
                fp4_quantizer_tied: torch.nn.Module,
                fp4_quantizer_SR: torch.nn.Module,
                HS: torch.Tensor
    ):  
        # Forward pass: No SR for inputs/weights (matching TE fp4_quant_fwd_inp/weight)
        fp4_quantiser_fwd = fp4_quantizer_tied
        
        if config.use_approx['use_hadamard'] in ['forward_exact', 'all_exact'] and HS is not None:
            A_ = (A.contiguous().view(-1, config.block_size) @ HS.T).view(A.shape)
            B_ = (B.t().contiguous().view(-1, config.block_size) @ HS.T).view(B.t().shape)
            A_mx_lp, _, _ = new_to_mx(A_, scalingModule, config.gemm_kernel_choice, fp4_quantiser_fwd)
            B_mx_lp, _, _ = new_to_mx(B_, scalingModule, config.gemm_kernel_choice, fp4_quantiser_fwd)
        else:
            A_mx_lp, _, _ = new_to_mx(A, scalingModule, config.gemm_kernel_choice, fp4_quantiser_fwd)
            B_mx_lp, _, _ = new_to_mx(B.t(), scalingModule, config.gemm_kernel_choice, fp4_quantiser_fwd)
        
        ctx.config = config
        ctx.quantgrad = quantgrad
        ctx.scalegrad = scalegrad
        ctx.tensorscalegrad = tensorscalegrad
        ctx.q_prime = q_prime
        ctx.scalingModule = scalingModule
        ctx.fp4_quantizer_tied = fp4_quantizer_tied
        ctx.fp4_quantizer_SR = fp4_quantizer_SR
        ctx.HS = HS
        ctx.save_for_backward(A, B)
        return mx_matmul_right_transpose(A_mx_lp, B_mx_lp)


    @staticmethod
    def backward(ctx, grad_output):
        config = ctx.config
        quantgrad = ctx.quantgrad
        scalegrad = ctx.scalegrad
        tensorscalegrad = ctx.tensorscalegrad
        q_prime = ctx.q_prime
        scalingModule = ctx.scalingModule
        fp4_quantizer_tied = ctx.fp4_quantizer_tied
        fp4_quantizer_SR = ctx.fp4_quantizer_SR
        HS_back = ctx.HS
        A, B = ctx.saved_tensors
        
        # Determine gradient quantizer (SR for grads, matching TE fp4_quant_bwd_grad)
        # SR modes that enable stochastic rounding on gradients
        sr_enabled_modes = ['all_exact', 'IntelFP4_exact', 'all_activation_exact', 'backward_exact']
        fp4_quantiser_grad = fp4_quantizer_SR if config.use_approx['SR'] in sr_enabled_modes else fp4_quantizer_tied
        
        # Weight and activation quantizers: no SR (matching TE)
        fp4_quantiser_weight = fp4_quantizer_tied
        fp4_quantiser_activation = fp4_quantizer_tied  # No SR for activations in backward
        
        if config.use_approx['use_hadamard'] in ['all_exact', 'backward_exact'] and HS_back is not None:
            # Apply Hadamard to gradients
            grad_output_H = (grad_output.reshape(-1, config.block_size) @ HS_back.T).reshape(grad_output.shape)
            grad_output_T = (grad_output.t().reshape(-1, config.block_size) @ HS_back.T).reshape(grad_output.t().shape)
            
            # Quantize gradients with SR (matching TE)
            grad_mx_lp, _, _ = new_to_mx(grad_output_H, scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_grad)
            grad_T_mx_lp, _, _ = new_to_mx(grad_output_T, scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_grad)
            
            # Apply Hadamard to weight (for dgrad)
            B_ = (B.reshape(-1, config.block_size) @ HS_back.T).reshape(B.shape)
            B_dim_mx_lp, _, B_scaled_hp_lp = new_to_mx(B_, scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_weight)
            
            # Apply Hadamard to A.t() (for wgrad) - no SR for activations
            A_ = (A.t().reshape(-1, config.block_size) @ HS_back.T).reshape(A.t().shape)
            A_dim_mx_lp, _, A_scaled_hp_lp = new_to_mx(A_, scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_activation)
        else:
            # Quantize gradients with SR (matching TE bwd_grad settings)
            grad_mx_lp, _, _ = new_to_mx(grad_output, scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_grad)
            grad_T_mx_lp, _, _ = new_to_mx(grad_output.t(), scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_grad)
            
            # Weight for dgrad: no SR 
            B_dim_mx_lp, _, B_scaled_hp_lp = new_to_mx(B, scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_weight)
            
            # A.t() for wgrad: no SR for activations (matching TE fp4_quant_fwd_inp)
            A_dim_mx_lp, _, A_scaled_hp_lp = new_to_mx(A.t(), scalingModule, config.gemm_kernel_choice, fp4_quantiser=fp4_quantiser_activation)
        
        # dgrad: dY @ B^T
        gradA = mx_matmul_right_transpose(grad_mx_lp, B_dim_mx_lp)
        
        # wgrad: A^T @ dY 
        gradB = mx_matmul_right_transpose(A_dim_mx_lp, grad_T_mx_lp)
        
        # For gradient adjustments
        A_dim_mx_lp_grad = A_dim_mx_lp
        B_dim_mx_lp_grad = B_dim_mx_lp

        # Apply quantization gradient corrections
        gradA = quantgrad(A_scaled_hp_lp.contiguous().t(), gradA)
        gradB = quantgrad(B_scaled_hp_lp.contiguous(), gradB)
        
        # Scale gradient adjustments
        gradA = scalegrad(gradA.t(), A_dim_mx_lp_grad, q_prime, A.t()).t()
        gradB = scalegrad(gradB, B_dim_mx_lp_grad, q_prime, B)

        if config.use_approx['use_tensor_scaling'] or config.scale_range_normalisation:
            gradA = tensorscalegrad(gradA.t(), A_dim_mx_lp_grad, A.t(), config.use_approx['tensor_scaling_grad_est']).t()
            gradB = tensorscalegrad(gradB, B_dim_mx_lp_grad, B, config.use_approx['tensor_scaling_grad_est'])
        
        return gradA, gradB, None, None, None, None, None, None, None, None, None


class MXLinearDimFused(torch.nn.Linear):
    """
    Linear layer with the compute happening in emulate MX. Currently the MX
    matmul is emulated since there is no hardware support yet. Activations,
    weights and grads are casted to MX and back to high precision for each
    matmul.

    Input, weight and grad_output can have each their own MX element dtype.
    """

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: Optional[MXLinearDimConfig] = MXLinearDimConfig(),
    ):
        # TODO(before land): remove this
        assert isinstance(config, MXLinearDimConfig)
        mod.__class__ = MXLinearDimFused
        mod.config = config
        mod.tensorscalegrad = TensorScalingGrad()
        mod.quantiser_FP4 = E2M1Quantizer()
        mod.quantiser_FP4_SR = E2M1QuantizerSR()
        e2m1range= np.array([
                    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,0,
                    0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0
                ], dtype=np.float32)
        e2m1range_tensor  = torch.from_numpy(e2m1range)
        if config.use_approx['stepGradient'] =='STE':
            mod.quantgrad = STEquantisationGrad()
        elif config.use_approx['stepGradient'] == 'baseline':
            mod.quantgrad = BaselineQuantisationGrad(
                                                    quant_values=e2m1range_tensor,
                                                    k=config.use_approx['k'], dtype=config.use_approx['dtype'],
                                                        lb=config.use_approx['lb'],
                                                        ub=config.use_approx['ub']
                                                     )
        elif config.use_approx['stepGradient'] == 'spline':
            q_class = QvalueQuantizer(e2m1range)
            
            mod.quantgrad = LinearSplineQuantisationGrad( 
                                                        quant_values=e2m1range_tensor,
                                                        quant_func=q_class.quantize,
                                                        x_test= np.linspace(-6,6,40),
                                                        dtype=config.use_approx['dtype'],
                                                         lb=config.use_approx['lb'],
                                                        ub=config.use_approx['ub']
                                                         )
        elif config.use_approx['stepGradient'] == 'sigmoid':
            mod.quantgrad = SigmoidQuantisationGrad( quant_values=e2m1range_tensor,
                                                    temperature=config.use_approx['temperature'],
                                                    dtype=config.use_approx['dtype'],
                                                    lb=config.use_approx['lb'],
                                                    ub=config.use_approx['ub']
                                                    )

        if config.use_approx['smooth'] =='STE': 
            mod.scalegrad = STEscaleGrad()
        elif config.use_approx['smooth'] =='absmax':
            if config.fp_scale_factor:
                mod.scalegrad = AbsMaxGradFPScale()
            else:
                mod.scalegrad = AbsMaxGrad()
        elif config.use_approx['smooth'] == 'softsoftmax' or config.use_approx['smooth'] == 'hardsoftmax':
            if config.fp_scale_factor:
                mod.scalegrad = SoftMaxGradFPScale()
            else:
                mod.scalegrad = SoftMaxGrad()

        scaleRange, x_test  = getScaleGrid(config)
        scaleRange_tensor  = torch.from_numpy(scaleRange)

        if config.use_approx['qGradient'] =='STE':
            mod.q_prime = STEquantisationGrad()
        elif config.use_approx['qGradient'] == 'baseline':
            mod.q_prime = BaselineQuantisationGrad(
                                                    quant_values=scaleRange_tensor,
                                                    k=config.use_approx['k'], dtype=config.use_approx['dtype'],
                                                        lb=config.use_approx['lb'],
                                                        ub=config.use_approx['ub']
                                                     )
        elif config.use_approx['qGradient'] == 'spline':
            q_class = QvalueQuantizer(scaleRange)
            mod.q_prime = LinearSplineQuantisationGrad( 
                                                        quant_values= scaleRange_tensor,
                                                        quant_func=q_class.quantize,
                                                        x_test= x_test,
                                                        dtype=config.use_approx['dtype'],
                                                         lb=config.use_approx['lb'],
                                                        ub=config.use_approx['ub']
                                                         )
        elif config.use_approx['qGradient'] == 'sigmoid':
            mod.q_prime = SigmoidQuantisationGrad( quant_values=scaleRange_tensor,
                                                    temperature=config.use_approx['temperature'],
                                                    dtype=config.use_approx['dtype'],
                                                    lb=config.use_approx['lb'],
                                                    ub=config.use_approx['ub']
                                                    )
        
        mod.scalingModule = MXFPscalingModule(
            elem_dtype=config.elem_dtype,
            block_size=config.block_size,
            roundMode = config.roundMode,
            scale_type=config.scale_type,
            use_approx=config.use_approx,
            fp_scale_factor = config.fp_scale_factor,
            nan_handling_mode= config.nan_handling_mode,
            scale_range_normalisation = config.scale_range_normalisation,
            strategy=config.strategy,
            use_fp32_scaling=config.use_fp32_scaling,
            )
        
        if config.use_approx.get('use_hadamard') in ['forward_exact', 'all_exact', 'backward_exact']:
            if mod.in_features >= config.block_size:
                 if mod.in_features % config.block_size != 0:
                     raise ValueError(f"Input feature dimension ({mod.in_features}) must be divisible by block_size ({config.block_size}) for Hadamard transform.")
                 if mod.weight.shape[1] % config.block_size != 0:
                     raise ValueError(f"Weight's input feature dimension ({mod.weight.shape[1]}) must be divisible by block_size ({config.block_size}) for Hadamard transform.")
                 
                 # Compute Hadamard Matrix (Deterministic)
                 # We cache this to avoid scipy in forward path
                 _dev = mod.weight.device
                 _dt = mod.weight.dtype
                 
                 # Use scipy hadamard (returns numpy float64) -> tensor
                 H_val = torch.tensor(hadamard(config.block_size), dtype=_dt, device=_dev) / (config.block_size**0.5)
                 mod.register_buffer("H_fixed", H_val)

        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x_shape = x.shape
        in_features = self.in_features
        out_features = self.out_features
        # Reshape x to 2D for matmul: (..., F_in) -> (N_flat, F_in)
        if len(original_x_shape) == 1: # Input is (F_in)
            # This case is less common for transformer layers but supported by nn.Linear
            x_reshaped = x.unsqueeze(0) # (1, F_in)
            output_prefix_shape = [] # For reshaping back to (F_out)
        elif len(original_x_shape) == 2: # Input is (N, F_in)
            x_reshaped = x # (N, F_in)
            output_prefix_shape = [original_x_shape[0]] # For reshaping back to (N, F_out)
        elif len(original_x_shape) == 3: # Input is (B, S, F_in)
            x_reshaped = x.reshape(-1, in_features) # (B*S, F_in)
            output_prefix_shape = [original_x_shape[0], original_x_shape[1]] # For (B,S,F_out)
        else:
            # For inputs with more than 3 dimensions, torch.nn.Linear flattens all dims before in_features
            x_reshaped = x.reshape(-1, in_features)
            # Collect all leading dimensions for reshaping the output
            output_prefix_shape = list(original_x_shape[:-1])


        # Handle autocast
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype(x.device.type if x.device.type != 'mps' else 'cpu')
            x_for_matmul = x_reshaped.to(autocast_dtype)
            w = self.weight.to(autocast_dtype)
        else:
            x_for_matmul = x_reshaped
            w = self.weight

        config = self.config
        HS = None # Hadamard sketch tensor
        HS = None # Hadamard sketch tensor
        if getattr(self, "H_fixed", None) is not None:
             # Retrieve fixed Hadamard matrix and cast to input dtype
             H_val = self.H_fixed.to(x_for_matmul.dtype)
             # Generate dynamic random signs in the same dtype
             S_val = torch.where(torch.randn(config.block_size, device=H_val.device, dtype=H_val.dtype) > 0, 
                                 torch.tensor(1.0, dtype=H_val.dtype, device=H_val.device), 
                                 torch.tensor(-1.0, dtype=H_val.dtype, device=H_val.device))
             # Combine
             HS = H_val.T * S_val
        y_flat = FusedMXFPMatMul.apply(
            x_for_matmul ,
            w.t() , # Pass (F_in, F_out)
            config,
            self.quantgrad,
            self.scalegrad,
            self.tensorscalegrad,
            self.q_prime,
            self.scalingModule,
            self.quantiser_FP4,
            self.quantiser_FP4_SR,
            HS
        )
        # y_flat will have shape (N_flat, F_out)

        # Reshape y_flat back to match original input's batch/sequence dimensions
        if not output_prefix_shape: # Original input was 1D (F_in)
            y = y_flat.squeeze(0) # Output (F_out)
        else:
            y = y_flat.view(*output_prefix_shape, out_features)

        if self.bias is not None:
            y = y + self.bias
        return y


def swap_linear_with_mx_linear_fused(
    model,
    *,
    config: Optional[MXLinearDimConfig] = None,
    filter_fn=None,
):
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXLinearDimFused.from_float(mod, config=config),
        combined_filter_fn,
    )

def swap_llama_mlp_to_mxlinear(
    model,
    config: Optional[MXLinearDimConfig] = None,
    include_lm_head: bool = False,
):
    def mlp_linear_filter_fn(mod, fqn: str) -> bool:
        mlp_keywords = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj","self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]
        if include_lm_head:
            mlp_keywords.append("lm_head")
        return isinstance(mod, torch.nn.Linear) and any(k in fqn for k in mlp_keywords)

    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXLinearDimFused.from_float(mod, config=config),
        mlp_linear_filter_fn,
    )