"""Knowledge Distillation bidirecional com IA paralela especializada."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BidirectionalDistillation(nn.Module):
    """Distila conhecimento entre modelo principal e modelo auxiliar."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def kl_divergence(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        student = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(student, teacher, reduction="batchmean") * (self.temperature**2)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
        student_to_teacher: bool = True,
    ) -> Tensor:
        hard_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        soft_loss = self.kl_divergence(student_logits, teacher_logits)
        if not student_to_teacher:
            soft_loss = soft_loss + self.kl_divergence(teacher_logits, student_logits)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss