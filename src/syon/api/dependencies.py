"""Dependências compartilhadas da API."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from syon.model import SyonModel


@lru_cache(maxsize=4)
def get_model(model_name: str) -> SyonModel:
    models_dir = Path(os.getenv("SYON_MODELS_DIR", "/models"))
    return SyonModel.from_config(model_name=model_name, models_dir=models_dir)