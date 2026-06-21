"""Interface de chat multi-turn."""

from __future__ import annotations

from typing import TypedDict

from syon.inference.engine import GenerationParams, InferenceEngine
from syon.security.guardrails import check_input, check_output, should_refuse_topic


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatSession:
    """Sessão de chat com histórico e guardrails."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def build_prompt(self, messages: list[ChatMessage]) -> str:
        parts = [
            "<|system|>Você é Syon, LLM especializado em programação e cybersegurança.",
        ]
        for message in messages:
            role = message["role"].strip().lower()
            content = message["content"]
            if role == "user":
                parts.append(f"<|user|>{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>{content}")
            elif role == "system":
                parts.append(f"<|system|>{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        if not messages:
            raise ValueError("messages não pode ser vazio")

        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"].lower() == "user"),
            "",
        )
        check_input(last_user)
        refusal = should_refuse_topic(last_user)
        if refusal:
            return refusal

        prompt = self.build_prompt(messages)
        response = self.engine.generate(
            prompt,
            GenerationParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p),
        )
        return check_output(response)