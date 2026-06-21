"""Schemas Pydantic da API REST."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = "syon-7b"
    prompt: str
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class CompletionChoice(BaseModel):
    text: str
    index: int = 0


class CompletionResponse(BaseModel):
    id: str
    model: str
    choices: list[CompletionChoice]
    usage: dict[str, int]


class ChatMessageSchema(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "syon-13b"
    messages: list[ChatMessageSchema]
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class ChatChoice(BaseModel):
    message: ChatMessageSchema
    index: int = 0


class ChatResponse(BaseModel):
    id: str
    model: str
    choices: list[ChatChoice]
    usage: dict[str, int]


class SecurityAnalysisRequest(BaseModel):
    model: str = "syon-7b"
    code: str
    language: str = "python"
    include_llm_analysis: bool = True


class SecurityAnalysisResponse(BaseModel):
    model: str
    language: str
    risk_level: str
    findings: list[dict[str, Any]]
    recommendations: list[str]
    owasp_categories: list[str]
    cvss_estimate: float | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None