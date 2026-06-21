import torch

from training.losses.contrastive import ContrastiveLoss
from training.losses.security_aware import SecurityAwareLoss


def test_security_aware_loss_computes():
    loss_fn = SecurityAwareLoss()
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    loss = loss_fn(logits, labels)
    assert loss.ndim == 0
    assert float(loss) > 0


def test_contrastive_loss_computes():
    loss_fn = ContrastiveLoss()
    anchor = torch.randn(4, 32)
    positive = torch.randn(4, 32)
    loss = loss_fn(anchor, positive)
    assert float(loss) >= 0