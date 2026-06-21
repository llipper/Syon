"""Utilitários de tratamento de exceções e retry."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def retry_with_backoff(
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator com retry exponencial e backoff."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            "Falha após %d tentativas em %s: %s",
                            max_retries,
                            func.__name__,
                            exc,
                        )
                        raise

                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        "Tentativa %d/%d falhou em %s (%s). Retry em %.2fs",
                        attempt,
                        max_retries,
                        func.__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper  # type: ignore[return-value]

    return decorator