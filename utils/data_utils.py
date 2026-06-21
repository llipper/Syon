"""Utilitários de carregamento e divisão de datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence


def load_dataset(path: str | Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    """Carrega dataset JSONL (uma amostra JSON por linha)."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

    records: list[dict[str, Any]] = []
    with file_path.open(encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido na linha {line_number} de {file_path}") from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"Cada linha do JSONL deve ser um objeto (linha {line_number})"
                )
            records.append(item)
    return records


def split_dataset(
    data: Sequence[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Divide dataset em conjuntos de treino, validação e teste."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"As proporções devem somar 1.0 (recebido: {total_ratio})")

    items = list(data)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = items[:train_end]
    val_set = items[train_end:val_end]
    test_set = items[val_end:]
    return train_set, val_set, test_set