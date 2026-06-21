"""Tratamento centralizado de erros da API."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.rest.models.response_models import ErrorResponse
from syon.exceptions import SecurityPolicyError, SyonError, UnsupportedLanguageError

logger = logging.getLogger("syon.api.errors")


def register_error_handlers(app: FastAPI) -> None:
    """Registra handlers globais de exceção no app FastAPI."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="http_error",
                detail=str(exc.detail),
                code=exc.status_code,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail=str(exc.errors()),
                code=422,
            ).model_dump(),
        )

    @app.exception_handler(SecurityPolicyError)
    async def security_policy_handler(request: Request, exc: SecurityPolicyError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="security_policy_violation", detail=str(exc), code=400).model_dump(),
        )

    @app.exception_handler(UnsupportedLanguageError)
    async def unsupported_language_handler(
        request: Request, exc: UnsupportedLanguageError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error="unsupported_language", detail=str(exc), code=400).model_dump(),
        )

    @app.exception_handler(SyonError)
    async def syon_error_handler(request: Request, exc: SyonError) -> JSONResponse:
        logger.exception("SyonError on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="syon_error", detail=str(exc), code=500).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="internal_error", detail="An unexpected error occurred", code=500).model_dump(),
        )