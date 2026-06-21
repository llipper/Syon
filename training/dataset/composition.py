"""Composição do dataset — 40% programação, 35% segurança, 15% complementar, 10% qualidade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from syon.config import CONFIGS_DIR


@dataclass
class DatasetSlice:
    name: str
    weight: float
    sources: list[str]


@dataclass
class DatasetComposition:
    slices: list[DatasetSlice]
    volume: dict[str, Any]

    def validate_weights(self) -> bool:
        total = sum(s.weight for s in self.slices)
        return abs(total - 1.0) < 1e-6

    def sample_budget(self, total_samples: int) -> dict[str, int]:
        return {s.name: int(total_samples * s.weight) for s in self.slices}


def load_composition(path: Path | None = None) -> DatasetComposition:
    config_path = path or (CONFIGS_DIR / "dataset" / "composition.yaml")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    composition = raw.get("composition", {})
    slices = [
        DatasetSlice(name=key, weight=float(data["weight"]), sources=list(data["sources"]))
        for key, data in composition.items()
    ]
    return DatasetComposition(slices=slices, volume=raw.get("volume", {}))