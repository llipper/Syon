"""Configuração da arquitetura Syon — transformer proprietário."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SyonModelConfig:
    """Hiperparâmetros do Syon 3 (decoder-only transformer proprietário)."""

    vocab_size: int = 8192
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 2048
    max_seq_length: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.1
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    model_type: str = "syon_3"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size deve ser divisível por num_heads")
        return self.hidden_size // self.num_heads

    def estimate_params(self) -> int:
        """Estimativa rápida de parâmetros treináveis."""
        emb = self.vocab_size * self.hidden_size
        per_layer = (
            4 * self.hidden_size * self.hidden_size  # QKV + out proj
            + 3 * self.hidden_size * self.intermediate_size  # SwiGLU
            + 4 * self.hidden_size  # norms
        )
        head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        return emb + self.num_layers * per_layer + head + self.hidden_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyonModelConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SyonModelConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: Path) -> SyonModelConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        arch = raw.get("architecture", raw)
        return cls.from_dict(arch)


PRESETS: dict[str, SyonModelConfig] = {
    "syon3": SyonModelConfig(
        vocab_size=8192,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        intermediate_size=1536,
        max_seq_length=512,
        dropout=0.1,
    ),
    "syon3_medium": SyonModelConfig(
        vocab_size=16384,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        max_seq_length=2048,
        dropout=0.1,
    ),
    "syon3_large": SyonModelConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        intermediate_size=11008,
        max_seq_length=4096,
        dropout=0.1,
    ),
}