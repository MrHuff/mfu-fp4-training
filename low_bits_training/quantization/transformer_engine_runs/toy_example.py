import transformer_engine.pytorch as te
import torch
from transformer_engine.common.recipe import Format, DelayedScaling, MXFP8BlockScaling, NVFP4BlockScaling




fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
mxfp8_format = Format.E4M3  # E4M3 used everywhere
mxfp8_recipe = MXFP8BlockScaling(fp8_format=mxfp8_format)
nvfp4_recipe = NVFP4BlockScaling()
torch.manual_seed(12345)

my_linear = te.Linear(768, 768, bias=True).bfloat16()

inp = torch.rand((1024, 768)).bfloat16().cuda()

with te.autocast(recipe=nvfp4_recipe):
    out = my_linear(inp)

print(out)

out.mean().backward()