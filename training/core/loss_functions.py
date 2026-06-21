"""Loss functions especializadas — migrado de training/losses/."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Re-export classes from legacy module for backward compatibility
from training.losses.contrastive import ContrastiveLoss
from training.losses.security_aware import SecurityAwareLoss


def cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Cross-entropy loss padrão."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


def security_aware_loss(
    logits: Tensor,
    labels: Tensor,
    security_mask: Tensor | None = None,
    weight: float = 0.15,
) -> Tensor:
    """Wrapper funcional para SecurityAwareLoss."""
    loss_fn = SecurityAwareLoss(weight=weight)
    return loss_fn(logits, labels, security_mask)


def code_quality_loss(logits: Tensor, labels: Tensor, quality_weights: Tensor | None = None) -> Tensor:
    """Penaliza tokens associados a baixa qualidade de código."""
    ce = cross_entropy_loss(logits, labels)
    if quality_weights is None:
        return ce
    probs = F.softmax(logits, dim=-1)
    penalty = (probs * quality_weights.unsqueeze(-1)).sum(dim=-1).mean()
    return ce + 0.1 * penalty


def custom_loss_weighted(
    losses: dict[str, Tensor],
    weights: dict[str, float],
) -> Tensor:
    """Combina múltiplas losses com pesos."""
    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for name, loss in losses.items():
        total = total + weights.get(name, 1.0) * loss
    return total


def perplexity_calculation(loss: Tensor) -> float:
    """Calcula perplexidade a partir da loss."""
    return float(math.exp(min(float(loss.item()), 20.0)))