"""Mixed precision training."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast


class MixedPrecisionTrainer:
    """Gerencia forward/backward em FP16 com gradient scaling."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = enabled if enabled is not None else torch.cuda.is_available()
        try:
            self.scaler = GradScaler("cuda", enabled=self.enabled)
            self._autocast_device = "cuda"
        except TypeError:
            self.scaler = GradScaler(enabled=self.enabled)
            self._autocast_device = None

    def forward_pass(self, model: torch.nn.Module, batch: dict[str, Tensor]) -> Any:
        """Forward pass com autocast FP16."""
        if self.enabled and self._autocast_device:
            context = autocast(self._autocast_device, enabled=True)
        elif self.enabled:
            context = autocast(enabled=True)
        else:
            context = nullcontext()
        with context:
            return model(**batch)

    def backward_pass(self, loss: Tensor) -> None:
        """Backward com gradient scaling."""
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Remove scaling dos gradientes antes do clip."""
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Step do optimizer com scaler."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()