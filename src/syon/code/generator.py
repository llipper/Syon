"""Geração de código especializada."""

from __future__ import annotations

from syon.code.languages import is_supported, normalize_language
from syon.exceptions import UnsupportedLanguageError
from syon.inference.engine import GenerationParams, InferenceEngine


class CodeGenerator:
    """Gera código a partir de especificações em linguagens suportadas."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def generate(
        self,
        specification: str,
        language: str = "python",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        lang = normalize_language(language)
        if not is_supported(lang):
            raise UnsupportedLanguageError(f"Linguagem não suportada: {language}")

        prompt = (
            f"<|system|>Você é Syon, especialista em programação e segurança.\n"
            f"Gere código {lang} seguro, idiomático e bem documentado.\n"
            f"<|user|>{specification}\n"
            f"<|assistant|>"
        )
        return self.engine.generate(
            prompt,
            GenerationParams(max_tokens=max_tokens, temperature=temperature, top_p=0.95),
        )