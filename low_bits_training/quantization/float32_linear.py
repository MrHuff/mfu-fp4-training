
import torch
import torch.nn as nn
import torch.nn.functional as F

class Float32Linear(nn.Linear):
    """
    A wrapper around nn.Linear that forces execution in Float32 precision,
    bypassing autocast if enabled. This is used to ensure bit-exact parity
    for sensitive layers (like the output head) that are prone to BF16 
    non-determinism.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check if autocast is enabled and disable component-wise
        # We handle device_type generic logic or specific to cuda
        device_type = input.device.type
        
        # Determine strict context
        # We manually cast input to float32 to ensure the operation happens in float32.
        # PyTorch addmm with Float32 Input + BFloat16 Weight -> Float32 Result (usually).
        # But to be safe, we disable autocast for this block.

        with torch.autocast(device_type=device_type, enabled=False):
            input_f32 = input if input.dtype == torch.float32 else input.to(torch.float32)
            weight_f32 = self.weight if self.weight.dtype == torch.float32 else self.weight.to(torch.float32)
            bias_f32 = None
            if self.bias is not None:
                bias_f32 = self.bias if self.bias.dtype == torch.float32 else self.bias.to(torch.float32)
            return F.linear(input_f32, weight_f32, bias_f32)
