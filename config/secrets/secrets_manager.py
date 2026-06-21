"""Gerenciador básico de secrets baseado em variáveis de ambiente."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

SECRETS_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = SECRETS_DIR / ".env"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse simples de arquivo .env sem dependências externas."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values

REQUIRED_SECRETS: tuple[str, ...] = (
    "SECRET_KEY",
)

OPTIONAL_SECRETS: tuple[str, ...] = (
    "WANDB_API_KEY",
    "REDIS_URL",
    "DATABASE_URL",
    "API_KEY",
)


class SecretsManager:
    """Carrega e valida secrets a partir de variáveis de ambiente."""

    def __init__(self, env_file: Path | None = None) -> None:
        self.env_file = env_file or DEFAULT_ENV_FILE
        self._cache: dict[str, str | None] = {}

    def load_env_file(self) -> dict[str, str]:
        """Carrega arquivo .env se existir."""
        return _parse_env_file(self.env_file)

    def get(self, key: str, default: str | None = None) -> str | None:
        """Obtém secret priorizando variáveis de ambiente do sistema."""
        if key in self._cache:
            return self._cache[key]

        value = os.environ.get(key)
        if value is None and self.env_file.exists():
            file_values = self.load_env_file()
            value = file_values.get(key)

        if value is None:
            value = default

        self._cache[key] = value
        return value

    def require(self, key: str) -> str:
        """Obtém secret obrigatório ou levanta erro."""
        value = self.get(key)
        if not value:
            raise KeyError(f"Secret obrigatório não configurado: {key}")
        return value

    def validate_required(self) -> list[str]:
        """Valida presença de secrets obrigatórios. Retorna lista de ausentes."""
        missing = [key for key in REQUIRED_SECRETS if not self.get(key)]
        return missing

    def as_dict(self, include_optional: bool = True) -> dict[str, Any]:
        """Exporta secrets como dicionário (valores mascarados para logs)."""
        keys = list(REQUIRED_SECRETS)
        if include_optional:
            keys.extend(OPTIONAL_SECRETS)

        result: dict[str, Any] = {}
        for key in keys:
            value = self.get(key)
            result[key] = self._mask(key, value) if value else None
        return result

    @staticmethod
    def _mask(key: str, value: str) -> str:
        if len(value) <= 4:
            return "****"
        return f"{value[:2]}***{value[-2:]}"


@lru_cache
def load_secrets(env_file: Path | None = None) -> SecretsManager:
    """Retorna instância singleton do gerenciador de secrets."""
    return SecretsManager(env_file=env_file)


def get_secret(key: str, default: str | None = None) -> str | None:
    """Atalho para obter um secret."""
    return load_secrets().get(key, default=default)