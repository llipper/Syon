"""Rota /v1/chat/completions — migrado de syon.api.routes.chat."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.rest.middleware.auth import verify_token
from api.rest.middleware.rate_limiting import check_rate_limit
from api.rest.models.request_models import ChatMessage, ChatRequest
from api.rest.models.response_models import ChatChoice, ChatResponse, ErrorResponse
from api.rest.utils import generate_request_id, get_model
from syon.exceptions import SecurityPolicyError, SyonError

router = APIRouter()


def manage_conversation_history(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Normaliza histórico de mensagens para o modelo."""
    return [{"role": m.role, "content": m.content} for m in messages]


def system_prompt_handling(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Garante que system prompt, se presente, está na primeira posição."""
    system_msgs = [m for m in messages if m.role == "system"]
    other_msgs = [m for m in messages if m.role != "system"]
    return system_msgs + other_msgs


def message_formatting(messages: list[ChatMessage]) -> str:
    """Formata mensagens como texto para logging/debug."""
    return "\n".join(f"{m.role}: {m.content[:100]}" for m in messages)


@router.post(
    "/chat/completions",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
)
def create_chat_completion(
    request: ChatRequest,
    _: str = Depends(verify_token),
    __: None = Depends(check_rate_limit),
) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    try:
        ordered = system_prompt_handling(request.messages)
        model = get_model(request.model)
        messages = manage_conversation_history(ordered)
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
        id=generate_request_id("chat"),
        model=request.model,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
        usage={
            "total_tokens": sum(len(m.content.split()) for m in request.messages) + len(text.split())
        },
    )