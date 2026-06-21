"""Gerenciamento de API keys."""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

AUTH_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "auth_config.yaml"


@dataclass
class APIKeyRecord:
    key_hash: str
    name: str
    created_at: str
    revoked: bool = False
    scopes: list[str] = field(default_factory=lambda: ["read", "write"])


class APIKeyManager:
    """Gerencia criação, validação e revogação de API keys."""

    def __init__(self):
        self._keys: dict[str, APIKeyRecord] = {}
        self._load_config_keys()

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _load_config_keys(self) -> None:
        env_key = os.getenv("SYON_API_KEY", "")
        if env_key:
            self._keys[self._hash_key(env_key)] = APIKeyRecord(
                key_hash=self._hash_key(env_key),
                name="env_default",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        if AUTH_CONFIG_PATH.exists():
            with AUTH_CONFIG_PATH.open(encoding="utf-8") as handle:
                config: dict[str, Any] = yaml.safe_load(handle) or {}
            for entry in config.get("api_keys", []):
                raw_key = entry.get("key", "")
                if raw_key:
                    self._keys[self._hash_key(raw_key)] = APIKeyRecord(
                        key_hash=self._hash_key(raw_key),
                        name=entry.get("name", "config_key"),
                        created_at=entry.get("created_at", datetime.now(timezone.utc).isoformat()),
                        scopes=entry.get("scopes", ["read", "write"]),
                    )

    def create_api_key(self, name: str, scopes: list[str] | None = None) -> tuple[str, APIKeyRecord]:
        """Gera nova API key e retorna (raw_key, record)."""
        raw_key = f"syon_{secrets.token_urlsafe(32)}"
        record = APIKeyRecord(
            key_hash=self._hash_key(raw_key),
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
            scopes=scopes or ["read", "write"],
        )
        self._keys[record.key_hash] = record
        return raw_key, record

    def validate_api_key(self, key: str) -> bool:
        """Valida se API key existe e não foi revogada."""
        record = self._keys.get(self._hash_key(key))
        return record is not None and not record.revoked

    def revoke_api_key(self, key: str) -> bool:
        """Revoga API key."""
        record = self._keys.get(self._hash_key(key))
        if record is None:
            return False
        record.revoked = True
        return True