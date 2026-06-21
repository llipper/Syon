"""Stub de servidor Flask para inferência Syon."""

from __future__ import annotations

from typing import Any


def create_flask_app() -> Any:
    """
    Stub Flask — use fastapi_server.create_app() em produção.

    Endpoints planejados:
    - POST /generate
    - POST /analyze
    - POST /chat
    - GET /health
    """
    raise NotImplementedError(
        "Flask server é um stub. Use inference.servers.fastapi_server.create_app()"
    )


def error_handler(error: Exception) -> tuple[dict[str, str], int]:
    return {"error": str(error)}, 500