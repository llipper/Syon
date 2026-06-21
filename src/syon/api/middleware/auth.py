"""Autenticação Bearer para API Syon."""

from __future__ import annotations

import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str:
    expected = os.getenv("SYON_API_KEY", "")
    if not expected:
        return "development"

    if credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials.credentials