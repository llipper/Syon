"""Rota de análise geral de código."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.rest.middleware.auth import verify_token
from api.rest.middleware.rate_limiting import check_rate_limit
from api.rest.models.request_models import CodeReviewRequest
from api.rest.models.response_models import ErrorResponse, SecurityReport
from api.rest.routes.security import detect_vulnerabilities, generate_security_report
from api.rest.utils import get_model
from syon.exceptions import SyonError, UnsupportedLanguageError

router = APIRouter()


def analyze_code(code: str, language: str) -> dict:
    """Análise básica de código."""
    lines = code.splitlines()
    return {
        "line_count": len(lines),
        "language": language,
        "has_functions": "def " in code or "function " in code,
    }


def review_code(code: str, language: str) -> list[str]:
    """Gera observações de code review."""
    notes: list[str] = []
    if len(code) > 10000:
        notes.append("Consider splitting into smaller modules")
    if "eval(" in code or "exec(" in code:
        notes.append("Avoid dynamic code execution (eval/exec)")
    if language == "python" and "import *" in code:
        notes.append("Avoid wildcard imports")
    return notes


def suggest_improvements(code: str, focus_areas: list[str]) -> list[str]:
    """Sugestões de melhoria baseadas em áreas de foco."""
    suggestions: list[str] = []
    if "security" in focus_areas:
        suggestions.append("Add input validation for external data")
        suggestions.append("Use parameterized queries for database access")
    if "quality" in focus_areas:
        suggestions.append("Add type hints and docstrings")
        suggestions.append("Increase test coverage")
    return suggestions


def generate_documentation(code: str, language: str) -> str:
    """Gera documentação resumida do código."""
    return f"Code analysis for {language}: {len(code.splitlines())} lines analyzed."


@router.post(
    "/analyze",
    response_model=SecurityReport,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def analyze_code_endpoint(
    request: CodeReviewRequest,
    _: str = Depends(verify_token),
    __: None = Depends(check_rate_limit),
) -> SecurityReport:
    try:
        stats = analyze_code(request.code, request.language)
        review_notes = review_code(request.code, request.language)
        suggestions = suggest_improvements(request.code, request.focus_areas) if request.include_suggestions else []
        doc = generate_documentation(request.code, request.language)

        try:
            model = get_model(request.model)
            llm_review = model.complete(
                f"Review this {request.language} code:\n{request.code[:2000]}",
                max_tokens=512,
                temperature=0.3,
            )
            review_notes.append(llm_review[:500])
        except (SyonError, FileNotFoundError):
            pass

        security_result = detect_vulnerabilities(request.code, request.language)
        report = generate_security_report(security_result)
        report.summary = f"{doc} | Lines: {stats['line_count']}"
        report.recommendations = list(dict.fromkeys(report.recommendations + suggestions + review_notes))
        return report
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SyonError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc