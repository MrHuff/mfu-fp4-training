
import torch
import numpy as np
import scipy.linalg

def get_hadamard_matrix(n: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Returns a normalized Hadamard matrix of size (n, n).
    n must be a power of 2.
    """
    # Check power of 2
    if (n & (n-1) != 0) or n == 0:
        raise ValueError(f"n must be a power of 2, got {n}")
        
    H = scipy.linalg.hadamard(n)
    H_tensor = torch.tensor(H, dtype=dtype, device=device)
    # Normalize
    H_tensor = H_tensor / (n**0.5)
    return H_tensor

def apply_rht(x: torch.Tensor, H: torch.Tensor, with_random_signs=True, seed=None) -> torch.Tensor:
    """
    Applies Randomized Hadamard Transform to x.
    x: Input tensor. Last dim must match H.shape[0].
    H: Normalized Hadamard matrix (N, N).
    
    Returns:
        x_transformed: (..., N)
    """
    if x.shape[-1] != H.shape[0]:
         raise ValueError(f"Last dimension of x ({x.shape[-1]}) must match H ({H.shape[0]})")
         
    # Generate random signs
    if with_random_signs:
        if seed is not None:
            gen = torch.Generator(device=x.device)
            gen.manual_seed(seed)
        else:
            gen = None
            
        signs = torch.randint(0, 2, (x.shape[-1],), device=x.device, generator=gen, dtype=x.dtype)
        # map 0 -> -1, 1 -> 1
        signs = 2 * signs - 1
        H_eff = H * signs.unsqueeze(0) # Equivalent to H @ diag(signs)
    else:
        H_eff = H

    # Apply x @ H_eff^T
    # If H is symmetric, H^T = H.
    # H is symmetric.
    # x @ (H * S)^T = x @ S @ H
    
    return torch.matmul(x, H_eff.T)

def inverse_rht(x: torch.Tensor, H: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
     """
     Inverse RHT. 
     If RHT was x @ S @ H, Inverse is x_hat @ H^T @ S.
     """
     # H is orthogonal: H @ H^T = I
     # S is involutory: S @ S = I
     H_eff = H * signs.unsqueeze(0)
     return torch.matmul(x, H_eff)
