"""FastAPI main app — migrado de syon.api.main."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.rest.middleware.error_handling import register_error_handlers
from api.rest.middleware.logging import LoggingMiddleware
from api.rest.routes import analysis, chat, completions, health, models, security
from api.rest.utils import load_api_config
from syon import __version__


def create_app() -> FastAPI:
    """Factory para criar instância configurada do FastAPI."""
    config = load_api_config()
    server_cfg = config.get("server", {})

    application = FastAPI(
        title="Syon API",
        description="API REST para programação e cybersegurança",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    cors_origins = os.getenv(
        "SYON_CORS_ORIGINS",
        ",".join(server_cfg.get("cors_origins", ["*"])),
    ).split(",")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(LoggingMiddleware)
    register_error_handlers(application)

    application.include_router(completions.router, prefix="/v1", tags=["completions"])
    application.include_router(chat.router, prefix="/v1", tags=["chat"])
    application.include_router(security.router, prefix="/v1", tags=["security"])
    application.include_router(analysis.router, prefix="/v1", tags=["analysis"])
    application.include_router(models.router, prefix="/v1", tags=["models"])
    application.include_router(health.router, tags=["health"])

    return application


app = create_app()


def run_server() -> None:
    """Inicia servidor uvicorn."""
    config = load_api_config()
    server_cfg = config.get("server", {})
    host = os.getenv("SYON_API_HOST", server_cfg.get("host", "0.0.0.0"))
    port = int(os.getenv("SYON_API_PORT", str(server_cfg.get("port", 8000))))
    workers = int(os.getenv("SYON_API_WORKERS", str(server_cfg.get("workers", 1))))
    uvicorn.run("api.rest.app:app", host=host, port=port, reload=False, workers=workers)


if __name__ == "__main__":
    run_server()