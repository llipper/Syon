"""Servidor FastAPI fino para inferência Syon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inference.core.inference_engine import GenerationParams, InferenceEngine
from inference.monitoring.health_check import health_status
from syon.config import SyonConfig


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95


class SecurityAnalysisRequest(BaseModel):
    code: str
    language: str = "python"


def create_app(engine: InferenceEngine | None = None) -> Any:
    """Cria aplicação FastAPI com endpoints de inferência."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Syon Inference Server", version="1.0.0")
    _engine = engine

    def get_engine() -> InferenceEngine:
        nonlocal _engine
        if _engine is None:
            config = SyonConfig.load()
            model_path = Path("models") / f"{config.model_name}.gguf"
            _engine = InferenceEngine(model_path, config)
            _engine.load()
        return _engine

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_status(get_engine())

    @app.get("/models")
    def list_models() -> dict[str, Any]:
        eng = get_engine()
        return {
            "models": [{"id": eng.config.model_name, "backend": eng.backend}],
        }

    @app.post("/completions")
    def completions(request: CompletionRequest) -> dict[str, Any]:
        eng = get_engine()
        params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        text = eng.generate(request.prompt, params)
        return {"text": text, "model": eng.config.model_name}

    @app.post("/chat/completions")
    def chat_completions(request: ChatRequest) -> dict[str, Any]:
        from security.input_validation.injection_detection import (
            check_input,
            should_refuse_topic,
        )
        from security.output_filtering.content_filter import check_output

        if not request.messages:
            raise HTTPException(status_code=400, detail="messages não pode ser vazio")

        last_user = next(
            (m.content for m in reversed(request.messages) if m.role.lower() == "user"),
            "",
        )
        check_input(last_user)
        refusal = should_refuse_topic(last_user)
        if refusal:
            return {"text": refusal, "refused": True}

        parts = ["<|system|>Você é Syon, especialista em programação e segurança."]
        for msg in request.messages:
            role = msg.role.strip().lower()
            if role == "user":
                parts.append(f"<|user|>{msg.content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>{msg.content}")
            elif role == "system":
                parts.append(f"<|system|>{msg.content}")
        parts.append("<|assistant|>")
        prompt = "\n".join(parts)

        eng = get_engine()
        params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        response = check_output(eng.generate(prompt, params))
        return {"text": response, "refused": False}

    @app.post("/security-analysis")
    def security_analysis(request: SecurityAnalysisRequest) -> dict[str, Any]:
        from security.vulnerability_scanning.sast import SecurityAnalyzer

        result = SecurityAnalyzer().analyze(request.code, language=request.language)
        return {
            "language": result.language,
            "risk_level": result.risk_level,
            "findings": [
                {
                    "cwe": f.cwe,
                    "severity": f.severity,
                    "description": f.description,
                    "line": f.line,
                }
                for f in result.findings
            ],
            "recommendations": result.recommendations,
            "owasp_categories": result.owasp_categories,
            "cvss_estimate": result.cvss_estimate,
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Inicia servidor uvicorn."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)