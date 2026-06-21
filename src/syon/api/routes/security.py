"""Endpoint /v1/security-analysis."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from syon.api.dependencies import get_model
from syon.api.middleware.auth import verify_api_key
from syon.api.schemas import SecurityAnalysisRequest, SecurityAnalysisResponse, ErrorResponse
from syon.exceptions import SyonError, UnsupportedLanguageError

router = APIRouter()


@router.post(
    "/security-analysis",
    response_model=SecurityAnalysisResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def analyze_security(
    request: SecurityAnalysisRequest,
    _: str = Depends(verify_api_key),
) -> SecurityAnalysisResponse:
    try:
        if request.include_llm_analysis:
            model = get_model(request.model)
            result = model.analyze_security(request.code, language=request.language)
        else:
            from syon.security.analyzer import SecurityAnalyzer

            result = SecurityAnalyzer().analyze(request.code, language=request.language)
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SyonError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError:
        from syon.security.analyzer import SecurityAnalyzer

        result = SecurityAnalyzer().analyze(request.code, language=request.language)

    return SecurityAnalysisResponse(
        model=request.model,
        language=result.language,
        risk_level=result.risk_level,
        findings=[asdict(f) for f in result.findings],
        recommendations=result.recommendations,
        owasp_categories=result.owasp_categories,
        cvss_estimate=result.cvss_estimate,
    )