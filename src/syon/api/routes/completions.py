"""Endpoint /v1/completions."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from syon.api.dependencies import get_model
from syon.api.middleware.auth import verify_api_key
from syon.api.schemas import CompletionRequest, CompletionResponse, CompletionChoice, ErrorResponse
from syon.exceptions import SecurityPolicyError, SyonError
from syon.model import SyonModel
from syon.security.guardrails import check_input

router = APIRouter()


@router.post(
    "/completions",
    response_model=CompletionResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def create_completion(
    request: CompletionRequest,
    _: str = Depends(verify_api_key),
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
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        model=request.model,
        choices=[CompletionChoice(text=text)],
        usage={"prompt_tokens": len(request.prompt.split()), "completion_tokens": len(text.split())},
    )