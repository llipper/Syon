"""Endpoint /v1/chat."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from syon.api.dependencies import get_model
from syon.api.middleware.auth import verify_api_key
from syon.api.schemas import ChatRequest, ChatResponse, ChatChoice, ChatMessageSchema, ErrorResponse
from syon.exceptions import SecurityPolicyError, SyonError

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def create_chat(
    request: ChatRequest,
    _: str = Depends(verify_api_key),
) -> ChatResponse:
    try:
        model = get_model(request.model)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        text = model.chat(
            messages,
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

    return ChatResponse(
        id=f"chat-{uuid.uuid4().hex[:12]}",
        model=request.model,
        choices=[ChatChoice(message=ChatMessageSchema(role="assistant", content=text))],
        usage={"total_tokens": sum(len(m.content.split()) for m in request.messages) + len(text.split())},
    )