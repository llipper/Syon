"""Samplers customizados."""

from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class WeightedSampler(Sampler[int]):
    """Sampler com pesos por domínio."""

    def __init__(self, dataset: Dataset, weights: list[float] | None = None):
        self.dataset = dataset
        n = len(dataset)
        self.weights = weights or [1.0] * n

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class DistributedSampler(Sampler[int]):
    """Sampler para treinamento distribuído."""

    def __init__(self, dataset: Dataset, rank: int = 0, world_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.rank = rank
        self.world_size = max(1, world_size)
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        indices = indices[self.rank :: self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset) // self.world_size + (1 if self.rank < len(self.dataset) % self.world_size else 0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class StratifiedSampler(Sampler[int]):
    """Sampler estratificado por label."""

    def __init__(self, dataset: Dataset, labels: list[str]):
        self.dataset = dataset
        self.label_indices: dict[str, list[int]] = {}
        for idx, label in enumerate(labels):
            self.label_indices.setdefault(label, []).append(idx)

    def __iter__(self) -> Iterator[int]:
        batches: list[int] = []
        max_len = max(len(v) for v in self.label_indices.values()) if self.label_indices else 0
        for i in range(max_len):
            for indices in self.label_indices.values():
                if i < len(indices):
                    batches.append(indices[i])
        random.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self.dataset)


def sample_batch(dataset: Dataset, batch_size: int) -> list:
    """Amostra batch aleatório."""
    indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
    return [dataset[i] for i in indices]