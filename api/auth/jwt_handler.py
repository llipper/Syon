"""JWT handling para autenticação da API."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

AUTH_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "auth_config.yaml"


def _load_auth_config() -> dict[str, Any]:
    if not AUTH_CONFIG_PATH.exists():
        return {}
    with AUTH_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class JWTHandler:
    """Cria e valida tokens JWT (implementação leve sem dependência externa)."""

    def __init__(self):
        config = _load_auth_config().get("jwt", {})
        self.secret_key = os.getenv("SYON_JWT_SECRET", config.get("secret_key", ""))
        self.algorithm = config.get("algorithm", "HS256")
        self.expire_minutes = int(config.get("expire_minutes", 60))

    def create_token(self, subject: str, extra_claims: dict[str, Any] | None = None) -> str:
        """Cria token JWT simplificado (header.payload.signature base64)."""
        import base64
        import hashlib
        import json

        now = datetime.now(timezone.utc)
        payload = {
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=self.expire_minutes)).timestamp()),
        }
        if extra_claims:
            payload.update(extra_claims)

        header = base64.urlsafe_b64encode(json.dumps({"alg": self.algorithm, "typ": "JWT"}).encode()).decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signature_input = f"{header}.{body}.{self.secret_key}"
        signature = hashlib.sha256(signature_input.encode()).hexdigest()[:32]
        return f"{header}.{body}.{signature}"

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verifica token e retorna payload ou None."""
        import base64
        import hashlib
        import json

        if not self.secret_key:
            return None

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header, body, signature = parts
            expected_sig = hashlib.sha256(f"{header}.{body}.{self.secret_key}".encode()).hexdigest()[:32]
            if signature != expected_sig:
                return None
            padded = body + "=" * (-len(body) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None

    def refresh_token(self, token: str) -> str | None:
        """Renova token se ainda válido."""
        payload = self.verify_token(token)
        if not payload:
            return None
        return self.create_token(payload.get("sub", "user"))