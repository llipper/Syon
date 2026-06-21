"""Rota /v1/completions — migrado de syon.api.routes.completions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.rest.middleware.auth import verify_token
from api.rest.middleware.rate_limiting import check_rate_limit
from api.rest.models.request_models import CompletionRequest
from api.rest.models.response_models import CompletionChoice, CompletionResponse, ErrorResponse
from api.rest.utils import generate_request_id, get_model
from syon.exceptions import SecurityPolicyError, SyonError
from syon.security.guardrails import check_input

router = APIRouter()


def validate_request(request: CompletionRequest) -> CompletionRequest:
    """Valida parâmetros da requisição de completion."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    return request


@router.post(
    "/completions",
    response_model=CompletionResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def create_completion(
    request: CompletionRequest = Depends(validate_request),
    _: str = Depends(verify_token),
    __: None = Depends(check_rate_limit),
) -> CompletionResponse:
    try:
        check_input(request.prompt)
        model = get_model(request.model)
        text = model.complete(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except SecurityPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SyonError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return CompletionResponse(
        id=generate_request_id("cmpl"),
        model=request.model,
        choices=[CompletionChoice(text=text)],
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(request.prompt.split()) + len(text.split()),
        },
    )