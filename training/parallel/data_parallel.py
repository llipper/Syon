"""Data parallelism strategy."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from training.parallel.utils import get_world_size, setup_torch_distributed


class DataParallelStrategy:
    """Estratégia de data parallelism com DDP."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.rank, self.world_size, self.local_rank = setup_torch_distributed(self.config)

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Envolve modelo com DDP se world_size > 1."""
        if self.world_size <= 1:
            return model
        device_ids = [self.local_rank] if torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids)

    def distribute_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Move batch para device correto."""
        device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def aggregate_losses(self, loss: Tensor) -> Tensor:
        """Agrega losses entre processos."""
        if self.world_size <= 1 or not torch.distributed.is_initialized():
            return loss
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        return loss / get_world_size()