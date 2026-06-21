"""Logging middleware para requisições da API."""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("syon.api")


def log_request(request: Request) -> None:
    """Registra detalhes da requisição recebida."""
    logger.info(
        "request method=%s path=%s client=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )


def log_response(request: Request, status_code: int, elapsed_ms: float) -> None:
    """Registra detalhes da resposta enviada."""
    logger.info(
        "response method=%s path=%s status=%d elapsed_ms=%.2f",
        request.method,
        request.url.path,
        status_code,
        elapsed_ms,
    )


def log_errors(request: Request, exc: Exception) -> None:
    """Registra erros durante processamento."""
    logger.error(
        "error method=%s path=%s error=%s",
        request.method,
        request.url.path,
        str(exc),
        exc_info=True,
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware que loga todas as requisições e respostas."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        log_request(request)
        try:
            response = await call_next(request)
        except Exception as exc:
            log_errors(request, exc)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_response(request, response.status_code, elapsed_ms)
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response