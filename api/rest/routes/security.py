"""Rota /v1/security-analysis — migrado de syon.api.routes.security."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException

from api.rest.middleware.auth import verify_token
from api.rest.middleware.rate_limiting import check_rate_limit
from api.rest.models.request_models import SecurityAnalysisRequest
from api.rest.models.response_models import ErrorResponse, SecurityAnalysisResponse, SecurityReport
from api.rest.utils import get_model
from syon.exceptions import SyonError, UnsupportedLanguageError

router = APIRouter()


def detect_vulnerabilities(code: str, language: str):
    """Executa análise estática de vulnerabilidades."""
    from syon.security.analyzer import SecurityAnalyzer

    return SecurityAnalyzer().analyze(code, language=language)


def generate_security_report(result) -> SecurityReport:
    """Converte resultado de análise em relatório estruturado."""
    return SecurityReport(
        summary=f"Risk level: {result.risk_level}",
        risk_level=result.risk_level,
        findings=[asdict(f) for f in result.findings],
        recommendations=result.recommendations,
        compliance_status={},
    )


def compliance_check(code: str, frameworks: list[str]) -> dict[str, str]:
    """Verificação básica de compliance por framework."""
    status = {}
    for framework in frameworks:
        status[framework] = "not_evaluated" if not code.strip() else "review_required"
    return status


@router.post(
    "/security-analysis",
    response_model=SecurityAnalysisResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def analyze_code_security(
    request: SecurityAnalysisRequest,
    _: str = Depends(verify_token),
    __: None = Depends(check_rate_limit),
) -> SecurityAnalysisResponse:
    try:
        if request.include_llm_analysis:
            model = get_model(request.model)
            result = model.analyze_security(request.code, language=request.language)
        else:
            result = detect_vulnerabilities(request.code, request.language)
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SyonError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError:
        result = detect_vulnerabilities(request.code, request.language)

    _ = generate_security_report(result)
    if request.compliance_frameworks:
        compliance_check(request.code, request.compliance_frameworks)

    return SecurityAnalysisResponse(
        model=request.model,
        language=result.language,
        risk_level=result.risk_level,
        findings=[asdict(f) for f in result.findings],
        recommendations=result.recommendations,
        owasp_categories=result.owasp_categories,
        cvss_estimate=result.cvss_estimate,
    )