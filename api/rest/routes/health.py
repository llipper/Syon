"""Rota /health — health checks."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter

from syon import __version__

router = APIRouter()

_start_time = time.time()


def health_check() -> dict[str, str]:
    """Verificação básica de saúde."""
    return {"status": "ok", "version": __version__}


def readiness_check() -> dict[str, str | bool]:
    """Verifica se o serviço está pronto para receber tráfego."""
    models_dir = os.getenv("SYON_MODELS_DIR", "/models")
    return {
        "status": "ready",
        "version": __version__,
        "models_dir_exists": os.path.isdir(models_dir),
    }


def detailed_status() -> dict:
    """Status detalhado do serviço."""
    uptime = time.time() - _start_time
    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": round(uptime, 2),
        "environment": os.getenv("SYON_ENV", "development"),
        "api_host": os.getenv("SYON_API_HOST", "0.0.0.0"),
    }


@router.get("/health")
def health_endpoint() -> dict[str, str]:
    return health_check()


@router.get("/health/ready")
def readiness_endpoint() -> dict[str, str | bool]:
    return readiness_check()


@router.get("/health/status")
def detailed_status_endpoint() -> dict:
    return detailed_status()