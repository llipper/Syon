"""Rota /v1/models — listagem de modelos disponíveis."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.rest.middleware.auth import verify_token
from api.rest.models.response_models import ErrorResponse, ModelInfo, ModelsListResponse

router = APIRouter()

AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="syon-7b",
        capabilities=["completion", "security-analysis", "code-generation"],
        max_context_length=4096,
    ),
    ModelInfo(
        id="syon-13b",
        capabilities=["chat", "completion", "security-analysis", "code-generation"],
        max_context_length=8192,
    ),
    ModelInfo(
        id="syon-70b",
        capabilities=["chat", "completion", "security-analysis", "code-generation", "reasoning"],
        max_context_length=8192,
    ),
]


def list_models() -> list[ModelInfo]:
    """Retorna lista de modelos registrados."""
    return AVAILABLE_MODELS


def get_model_info(model_id: str) -> ModelInfo:
    """Retorna informações de um modelo específico."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


def get_model_capabilities(model_id: str) -> list[str]:
    """Retorna capacidades de um modelo."""
    return get_model_info(model_id).capabilities


@router.get(
    "/models",
    response_model=ModelsListResponse,
    responses={401: {"model": ErrorResponse}},
)
def list_available_models(_: str = Depends(verify_token)) -> ModelsListResponse:
    return ModelsListResponse(data=list_models())


@router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def get_model_details(model_id: str, _: str = Depends(verify_token)) -> ModelInfo:
    return get_model_info(model_id)