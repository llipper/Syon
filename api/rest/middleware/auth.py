"""Autenticação da API — migrado de syon.api.middleware.auth."""

from __future__ import annotations

import os
from typing import Callable

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from api.auth.api_keys import APIKeyManager
from api.auth.jwt_handler import JWTHandler

_bearer = HTTPBearer(auto_error=False)
_api_key_manager = APIKeyManager()
_jwt_handler = JWTHandler()


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str:
    """Verifica Bearer token (API key ou JWT)."""
    expected_key = os.getenv("SYON_API_KEY", "")
    if not expected_key and not _jwt_handler.secret_key:
        return "development"

    if credentials is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    token = credentials.credentials

    if expected_key and token == expected_key:
        return token

    if _api_key_manager.validate_api_key(token):
        return token

    payload = _jwt_handler.verify_token(token)
    if payload:
        return payload.get("sub", token)

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


class APIKeyAuth:
    """Dependency wrapper para autenticação por API key."""

    def __call__(
        self,
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> str:
        return verify_token(credentials)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Middleware opcional para validar JWT em todas as rotas protegidas."""

    def __init__(self, app, exempt_paths: list[str] | None = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            if os.getenv("SYON_API_KEY"):
                raise HTTPException(status_code=401, detail="Authorization required")
            return await call_next(request)

        token = auth_header.removeprefix("Bearer ").strip()
        if not (_jwt_handler.verify_token(token) or APIKeyManager().validate_api_key(token)):
            raise HTTPException(status_code=401, detail="Invalid token")

        return await call_next(request)