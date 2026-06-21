"""Utilitários de medição de tempo."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def measure_elapsed_time() -> Generator[dict[str, float], None, None]:
    """Context manager que mede tempo decorrido em segundos e milissegundos."""
    metrics: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield metrics
    finally:
        elapsed = time.perf_counter() - start
        metrics["elapsed_seconds"] = elapsed
        metrics["elapsed_ms"] = elapsed * 1000.0


def timed(func: F) -> F:
    """Decorator que registra tempo de execução no atributo last_elapsed_ms."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with measure_elapsed_time() as metrics:
            result = func(*args, **kwargs)
        wrapper.last_elapsed_ms = metrics["elapsed_ms"]  # type: ignore[attr-defined]
        return result

    wrapper.last_elapsed_ms = 0.0  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]