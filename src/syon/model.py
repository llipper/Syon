"""Classe principal SyonModel — API pública do SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from syon.chat import ChatMessage, ChatSession
from syon.code.generator import CodeGenerator
from syon.config import SyonConfig
from syon.inference.engine import InferenceEngine
from syon.security.analyzer import SecurityAnalysisResult, SecurityAnalyzer


class SyonModel:
    """Modelo Syon para inferência local (GGUF/FP16)."""

    def __init__(
        self,
        engine: InferenceEngine,
        config: SyonConfig,
        *,
        _skip_load: bool = False,
    ):
        self.config = config
        self.engine = engine
        self._security = SecurityAnalyzer()
        self._code = CodeGenerator(engine)
        self._chat = ChatSession(engine)
        if not _skip_load:
            self.engine.load()

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        model_name: str | None = None,
    ) -> SyonModel:
        path = Path(model_path)
        inferred_name = model_name or path.stem.replace("-fp16", "").replace(".gguf", "")
        config = SyonConfig.load(inferred_name)
        engine = InferenceEngine(path, config)
        return cls(engine, config)

    @classmethod
    def from_config(cls, model_name: str = "syon-7b", models_dir: str | Path = "models") -> SyonModel:
        """Carrega modelo a partir do nome e diretório padrão."""
        config = SyonConfig.load(model_name)
        dist = config.model_config.get("distribution", {})
        gguf_name = dist.get("gguf", f"{model_name}.gguf")
        model_path = Path(models_dir) / gguf_name
        return cls.load(model_path, model_name=model_name)

    def analyze_security(self, code: str, language: str = "python") -> SecurityAnalysisResult:
        static = self._security.analyze(code, language=language)
        if self.engine.is_loaded:
            prompt = (
                f"<|system|>Analise vulnerabilidades de segurança no código {language}.\n"
                f"<|user|>```{language}\n{code}\n```\n"
                f"<|assistant|>"
            )
            _ = self.engine.generate(prompt)
        return static

    def generate_code(
        self,
        specification: str,
        language: str = "python",
        **kwargs: Any,
    ) -> str:
        return self._code.generate(specification, language=language, **kwargs)

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        return self._chat.chat(messages, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        from syon.inference.engine import GenerationParams

        params = GenerationParams(
            max_tokens=int(kwargs.get("max_tokens", 2048)),
            temperature=float(kwargs.get("temperature", self.config.default_temperature)),
            top_p=float(kwargs.get("top_p", self.config.default_top_p)),
        )
        return self.engine.generate(prompt, params)