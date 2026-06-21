"""Health checks para serviços de inferência."""

from __future__ import annotations

from typing import Any

from inference.hardware.gpu import detect_gpu


def check_model_loaded(engine: Any) -> dict[str, Any]:
    loaded = getattr(engine, "is_loaded", False)
    backend = getattr(engine, "backend", "unknown")
    return {"model_loaded": bool(loaded), "backend": backend}


def check_gpu_health() -> dict[str, Any]:
    info = detect_gpu()
    if not info["available"]:
        return {"healthy": True, "gpu": False, "message": "CPU-only mode"}
    return {
        "healthy": True,
        "gpu": True,
        "device_count": info["count"],
        "devices": info["devices"],
    }


def check_memory_health(threshold_percent: float = 90.0) -> dict[str, Any]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        used_pct = mem.percent
        return {
            "healthy": used_pct < threshold_percent,
            "used_percent": used_pct,
            "available_gb": round(mem.available / (1024**3), 2),
        }
    except ImportError:
        return {"healthy": True, "message": "psutil não disponível"}


def health_status(engine: Any) -> dict[str, Any]:
    """Status agregado de saúde do serviço."""
    model = check_model_loaded(engine)
    gpu = check_gpu_health()
    memory = check_memory_health()
    healthy = model["model_loaded"] and memory.get("healthy", True)
    return {
        "status": "healthy" if healthy else "degraded",
        "model": model,
        "gpu": gpu,
        "memory": memory,
    }