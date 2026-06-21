"""Middlewares da API REST."""

from api.rest.middleware.auth import APIKeyAuth, JWTAuthMiddleware, verify_token
from api.rest.middleware.error_handling import register_error_handlers
from api.rest.middleware.logging import LoggingMiddleware
from api.rest.middleware.rate_limiting import RateLimiter, check_rate_limit

__all__ = [
    "APIKeyAuth",
    "JWTAuthMiddleware",
    "verify_token",
    "register_error_handlers",
    "LoggingMiddleware",
    "RateLimiter",
    "check_rate_limit",
]