"""Security-Aware Loss — penaliza outputs inseguros durante treinamento."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SecurityAwareLoss(nn.Module):
    """Combina cross-entropy com penalidade para tokens de risco de segurança."""

    RISK_TOKEN_IDS: set[int] = set()

    def __init__(self, weight: float = 0.15, ignore_index: int = -100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        security_mask: Tensor | None = None,
    ) -> Tensor:
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )
        if security_mask is None:
            return ce

        probs = F.softmax(logits, dim=-1)
        risk_penalty = (probs * security_mask.unsqueeze(-1)).sum(dim=-1).mean()
        return ce + self.weight * risk_penalty