"""Utilitários de manipulação de arquivos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: str | Path, *, encoding: str = "utf-8") -> Any:
    """Carrega conteúdo JSON de um arquivo."""
    file_path = Path(path)
    with file_path.open(encoding=encoding) as handle:
        return json.load(handle)


def save_json(
    path: str | Path,
    data: Any,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Salva dados em formato JSON."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding=encoding) as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=ensure_ascii)
    return file_path


def load_yaml(path: str | Path, *, encoding: str = "utf-8") -> dict[str, Any]:
    """Carrega conteúdo YAML de um arquivo."""
    file_path = Path(path)
    with file_path.open(encoding=encoding) as handle:
        content = yaml.safe_load(handle)
    return content if isinstance(content, dict) else {}