"""Contrastive Learning Loss para treinamento paralelo."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContrastiveLoss(nn.Module):
    """InfoNCE entre representações do modelo principal e auxiliar."""

    def __init__(self, temperature: float = 0.07, weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, anchor: Tensor, positive: Tensor) -> Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = torch.matmul(anchor, positive.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return self.weight * loss