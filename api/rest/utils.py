"""Utilitários da API REST — migrado de syon.api.dependencies e schemas."""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from syon.model import SyonModel

API_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "api_config.yaml"


@lru_cache(maxsize=4)
def get_model(model_name: str) -> SyonModel:
    """Carrega e cacheia instância do modelo Syon."""
    models_dir = Path(os.getenv("SYON_MODELS_DIR", "/models"))
    return SyonModel.from_config(model_name=model_name, models_dir=models_dir)


def load_api_config() -> dict[str, Any]:
    """Carrega configuração da API a partir de YAML."""
    if not API_CONFIG_PATH.exists():
        return {}
    with API_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def generate_request_id(prefix: str = "req") -> str:
    """Gera ID único para requisições."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def format_response(data: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
    """Formata resposta padronizada com metadados opcionais."""
    response = dict(data)
    if request_id:
        response["request_id"] = request_id
    return response


def validate_parameters(params: dict[str, Any], rules: dict[str, tuple[Any, Any]]) -> list[str]:
    """Valida parâmetros numéricos contra limites min/max."""
    errors: list[str] = []
    for key, (min_val, max_val) in rules.items():
        if key not in params:
            continue
        value = params[key]
        if value < min_val or value > max_val:
            errors.append(f"{key} must be between {min_val} and {max_val}")
    return errors


def handle_exceptions(exc: Exception) -> dict[str, str]:
    """Converte exceção em payload de erro."""
    return {"error": type(exc).__name__, "detail": str(exc)}