"""AdamW com warm restarts conforme README."""

from __future__ import annotations

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def build_optimizer(model, config: dict):
    training = config.get("training", {}).get("optimizer", {})
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 2e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
    return optimizer, scheduler