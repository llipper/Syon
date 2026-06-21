"""Distributed training utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from training.parallel.utils import get_world_size, setup_torch_distributed


class DistributedTrainer:
    """Wrapper para treinamento distribuído com DDP."""

    def __init__(self, model: torch.nn.Module, config: dict[str, Any]):
        self.config = config
        self.rank, self.world_size, self.local_rank = setup_torch_distributed(config)
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)

    def setup_ddp(self) -> None:
        """Inicializa processo distribuído se necessário."""
        if not dist.is_initialized():
            setup_torch_distributed(self.config)

    def sync_gradients(self) -> None:
        """Sincroniza gradientes entre processos."""
        if not dist.is_initialized() or self.world_size <= 1:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)

    def gather_results(self, tensor: Tensor) -> Tensor | None:
        """Coleta tensores de todos os ranks."""
        return gather_results(tensor, self.world_size)


def setup_ddp(config: dict[str, Any] | None = None) -> tuple[int, int, int]:
    """Inicializa DDP e retorna (rank, world_size, local_rank)."""
    return setup_torch_distributed(config)


def sync_gradients(model: torch.nn.Module, world_size: int = 1) -> None:
    """Sincroniza gradientes entre processos."""
    if not dist.is_initialized() or world_size <= 1:
        return
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def gather_results(tensor: Tensor, world_size: int = 1) -> Tensor | None:
    """Coleta tensores de todos os ranks."""
    if not dist.is_initialized() or world_size <= 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered).mean(dim=0)