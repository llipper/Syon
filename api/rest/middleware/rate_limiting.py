"""Rate limiting para API Syon."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from fastapi import HTTPException, Request

RATE_LIMIT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "rate_limit_config.yaml"


def _load_rate_limit_config() -> dict[str, Any]:
    if not RATE_LIMIT_CONFIG_PATH.exists():
        return {"default": {"requests_per_minute": 60, "burst": 10}}
    with RATE_LIMIT_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class RateLimiter:
    """Token bucket rate limiter por cliente (IP ou API key)."""

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _prune(self, key: str, now: float) -> None:
        window_start = now - 60.0
        self._buckets[key] = [t for t in self._buckets[key] if t > window_start]

    def check_rate_limit(self, client_id: str) -> tuple[bool, int]:
        """Retorna (allowed, remaining_requests)."""
        now = time.monotonic()
        with self._lock:
            self._prune(client_id, now)
            timestamps = self._buckets[client_id]
            if len(timestamps) >= self.requests_per_minute:
                return False, 0
            timestamps.append(now)
            remaining = max(0, self.requests_per_minute - len(timestamps))
            return True, remaining

    def get_remaining_requests(self, client_id: str) -> int:
        now = time.monotonic()
        with self._lock:
            self._prune(client_id, now)
            return max(0, self.requests_per_minute - len(self._buckets[client_id]))


_config = _load_rate_limit_config()
_default_cfg = _config.get("default", {})
_limiter = RateLimiter(
    requests_per_minute=int(_default_cfg.get("requests_per_minute", 60)),
    burst=int(_default_cfg.get("burst", 10)),
)


def check_rate_limit(request: Request) -> None:
    """Dependency que bloqueia requisições acima do limite."""
    client_id = request.headers.get("X-API-Key") or request.client.host if request.client else "unknown"
    allowed, remaining = _limiter.check_rate_limit(client_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )