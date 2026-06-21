"""Paralelismo de treinamento."""

from training.parallel.data_parallel import DataParallelStrategy
from training.parallel.utils import get_world_size, setup_torch_distributed, setup_xla_distributed

__all__ = [
    "DataParallelStrategy",
    "setup_torch_distributed",
    "setup_xla_distributed",
    "get_world_size",
]