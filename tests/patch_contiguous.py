import torch
import sys
import inspect

_orig_contiguous = torch.Tensor.contiguous

def _patched_contiguous(self, *args, **kwargs):
    strides = self.stride()
    shape = self.shape
    # If not contiguous, it will copy
    if not self.is_contiguous():
        print(f"CONTIGUOUS COPY: shape={shape}, strides={strides}, dtype={self.dtype}, device={self.device}", flush=True)
        stack = inspect.stack()
        for i, frame in enumerate(stack):
            print(f"  Frame {i}: {frame.filename}:{frame.lineno} in {frame.function}", flush=True)
    return _orig_contiguous(self, *args, **kwargs)

torch.Tensor.contiguous = _patched_contiguous

print("Patch applied for contiguous()")
