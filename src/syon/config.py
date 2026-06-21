"""Carregamento de configurações YAML do projeto Syon."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class SyonConfig:
    """Configuração agregada do Syon."""

    model_name: str = "syon-7b"
    model_config: dict[str, Any] = field(default_factory=dict)
    inference_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    dataset_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_name: str = "syon-7b",
        configs_dir: Path | None = None,
    ) -> SyonConfig:
        base = configs_dir or CONFIGS_DIR
        model_path = base / "model" / f"{model_name}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Configuração de modelo não encontrada: {model_path}")

        return cls(
            model_name=model_name,
            model_config=_load_yaml(model_path),
            inference_config=_load_yaml(base / "inference" / "default.yaml"),
            training_config=_load_yaml(base / "training" / "parallel_sft.yaml"),
            dataset_config=_load_yaml(base / "dataset" / "composition.yaml"),
        )

    @property
    def context_window(self) -> int:
        return int(self.model_config.get("architecture", {}).get("context_window", 8192))

    @property
    def default_temperature(self) -> float:
        return float(self.inference_config.get("inference", {}).get("temperature", 0.7))

    @property
    def default_top_p(self) -> float:
        return float(self.inference_config.get("inference", {}).get("top_p", 0.95))