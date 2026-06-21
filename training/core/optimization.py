"""Otimização — AdamW com schedule — migrado de training/optimizer.py."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR


class AdamWScheduler:
    """AdamW com warmup linear e cosine annealing."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        total_steps: int = 100000,
    ):
        training = config.get("training", {}).get("optimizer", {})
        self.learning_rate = float(training.get("learning_rate", 2e-5))
        self.weight_decay = float(training.get("weight_decay", 0.01))
        self.warmup_steps = int(training.get("warmup_steps", 2000))
        self.total_steps = total_steps

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> LambdaLR:
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return linear_warmup(step, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return cosine_annealing(progress)

        return LambdaLR(self.optimizer, lr_lambda)

    def step(self) -> None:
        self.scheduler.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def linear_warmup(step: int, warmup_steps: int) -> float:
    """Warmup linear de 0 a 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_annealing(progress: float, min_lr_ratio: float = 0.1) -> float:
    """Annealing cosseno entre 1.0 e min_lr_ratio."""
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def update_lr(optimizer: AdamW, new_lr: float) -> None:
    """Atualiza learning rate do optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def build_optimizer(model, config: dict[str, Any]):
    """Compatibilidade com training.optimizer.build_optimizer."""
    training = config.get("training", {}).get("optimizer", {})
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    return optimizer, scheduler