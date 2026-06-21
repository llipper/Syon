"""Custom dataloaders — integrado com training/dataset/curator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from training.data_pipeline.collate import prepare_batch
from training.data_pipeline.sampler import DistributedSampler, WeightedSampler
from training.dataset.curator import CuratedSample, DatasetCurator
from training.dataset.composition import load_composition


class CodeDataset(Dataset):
    """Dataset para amostras de código."""

    def __init__(self, samples: list[CuratedSample], max_length: int = 4096):
        self.samples = [s for s in samples if s.domain == "programming"]
        if not self.samples:
            self.samples = samples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        tokens = sample.text[: self.max_length]
        return {
            "text": tokens,
            "domain": sample.domain,
            "source": sample.source,
            "metadata": sample.metadata,
        }


class SecurityDataset(Dataset):
    """Dataset para amostras de segurança."""

    def __init__(self, samples: list[CuratedSample], max_length: int = 4096):
        self.samples = [s for s in samples if s.domain in ("cybersecurity", "security")]
        if not self.samples:
            self.samples = samples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {
            "text": sample.text[: self.max_length],
            "domain": sample.domain,
            "source": sample.source,
            "metadata": sample.metadata,
            "security_label": sample.metadata.get("cwe", "unknown"),
        }


class MergedDataset(Dataset):
    """Dataset combinando código e segurança."""

    def __init__(self, code_dataset: CodeDataset, security_dataset: SecurityDataset):
        self.code = code_dataset
        self.security = security_dataset

    def __len__(self) -> int:
        return len(self.code) + len(self.security)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < len(self.code):
            return self.code[idx]
        return self.security[idx - len(self.code)]


def get_batch(dataset: Dataset, indices: list[int]) -> list[dict[str, Any]]:
    """Obtém batch de índices."""
    return [dataset[i] for i in indices]


def create_dataloader(
    data_dir: Path,
    batch_size: int = 8,
    domain: str = "merged",
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """Cria DataLoader a partir de dados curados."""
    curator = DatasetCurator(data_dir, load_composition())
    samples = curator.curate_all()

    if domain == "code":
        dataset: Dataset = CodeDataset(samples)
    elif domain == "security":
        dataset = SecurityDataset(samples)
    else:
        code_ds = CodeDataset(samples)
        sec_ds = SecurityDataset(samples)
        dataset = MergedDataset(code_ds, sec_ds)

    sampler = DistributedSampler(dataset) if distributed else WeightedSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=prepare_batch,
    )