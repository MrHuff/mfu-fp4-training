#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch


class BF16LossScaler:
    def __init__(
        self,
        init_scale=2**10,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, model):
        self._found_inf = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    self._found_inf = True
                    break
        if not self._found_inf:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.scale)

    def step(self, optimizer):
        if self._found_inf:
            # Skip optimizer step if inf/nan found
            print(
                f"[LossScaler] Found inf/nan in gradients. Skipping step. Reducing scale from {self.scale}."
            )
            self.scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            optimizer.step()
            self._growth_tracker += 1
            if self._growth_tracker % self.growth_interval == 0:
                self.scale *= self.growth_factor
                print(f"[LossScaler] Increasing scale to {self.scale}")
        optimizer.zero_grad()
