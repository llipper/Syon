"""Utilidades de paralelismo."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def setup_torch_distributed(config: dict[str, Any] | None = None) -> tuple[int, int, int]:
    """
    Inicializa PyTorch distributed.
    Retorna (rank, world_size, local_rank).
    """
    config = config or {}
    dist_cfg = config.get("distributed_params", config.get("training", {}).get("distributed_params", {}))

    rank = int(os.getenv("RANK", dist_cfg.get("rank", 0)))
    world_size = int(os.getenv("WORLD_SIZE", dist_cfg.get("world_size", 1)))
    local_rank = int(os.getenv("LOCAL_RANK", dist_cfg.get("local_rank", 0)))

    if world_size > 1 and not dist.is_initialized():
        backend = dist_cfg.get("backend", "nccl" if torch.cuda.is_available() else "gloo")
        init_method = os.getenv("MASTER_ADDR", "env://")
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def setup_xla_distributed() -> tuple[int, int]:
    """Stub para setup XLA/TPU distributed."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        return xm.get_ordinal(), xm.world_size()
    except ImportError:
        return 0, 1


def get_world_size() -> int:
    """Retorna world size do processo distribuído."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1