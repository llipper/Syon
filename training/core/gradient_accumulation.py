"""Gradient accumulation utilities."""

from __future__ import annotations

from torch import Tensor
from torch.optim import Optimizer


def accumulate_gradients(step: int, accumulation_steps: int) -> bool:
    """Retorna True se deve executar optimizer step neste step."""
    return (step + 1) % accumulation_steps == 0


def effective_batch_size(batch_size: int, accumulation_steps: int, world_size: int = 1) -> int:
    """Calcula batch size efetivo."""
    return batch_size * accumulation_steps * world_size


def update_with_accumulation(
    optimizer: Optimizer,
    loss: Tensor,
    step: int,
    accumulation_steps: int,
    scaler_step_fn=None,
) -> bool:
    """
    Acumula gradientes e atualiza pesos quando atingir accumulation_steps.
    Retorna True se optimizer step foi executado.
    """
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()

    if accumulate_gradients(step, accumulation_steps):
        if scaler_step_fn:
            scaler_step_fn(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        return True
    return False