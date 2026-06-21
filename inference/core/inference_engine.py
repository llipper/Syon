"""Motor de inferência Syon — geração, batch e streaming."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.core.model_loader import LoadedModel, ModelLoader
from inference.core.post_processing import clean_output
from inference.monitoring.metrics import InferenceMetrics
from syon.config import SyonConfig
from syon.exceptions import InferenceError


@dataclass
class GenerationParams:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[str] | None = None


class InferenceEngine:
    """Encapsula carregamento e geração via GGUF, HuggingFace ou ONNX."""

    def __init__(self, model_path: str | Path, config: SyonConfig | None = None):
        self.model_path = Path(model_path)
        self.config = config or SyonConfig.load()
        self._loader = ModelLoader(self.model_path, self.config)
        self._loaded: LoadedModel | None = None
        self._metrics = InferenceMetrics()

    def load(self) -> None:
        self._loaded = self._loader.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def backend(self) -> str:
        return self._loaded.backend if self._loaded else "none"

    def _ensure_loaded(self) -> LoadedModel:
        if not self.is_loaded:
            self.load()
        assert self._loaded is not None
        return self._loaded

    def generate(self, prompt: str, params: GenerationParams | None = None) -> str:
        loaded = self._ensure_loaded()
        params = params or GenerationParams(
            temperature=self.config.default_temperature,
            top_p=self.config.default_top_p,
        )

        start = time.perf_counter()
        try:
            if loaded.backend == "gguf":
                text = self._generate_gguf(loaded, prompt, params)
            elif loaded.backend == "hf":
                text = self._generate_hf(loaded, prompt, params)
            elif loaded.backend == "onnx":
                text = self._generate_onnx_stub(loaded, prompt, params)
            else:
                raise InferenceError(f"Backend desconhecido: {loaded.backend}")
        except Exception as exc:
            self._metrics.log_errors(str(exc))
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._metrics.log_inference_time(elapsed_ms)
        self._metrics.log_token_throughput(len(text.split()), elapsed_ms)
        return clean_output(text)

    def batch_generate(
        self,
        prompts: list[str],
        params: GenerationParams | None = None,
    ) -> list[str]:
        """Gera respostas para múltiplos prompts sequencialmente."""
        if not prompts:
            return []
        return [self.generate(prompt, params) for prompt in prompts]

    def streaming_generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> Iterator[str]:
        """Gera tokens em streaming quando suportado pelo backend."""
        loaded = self._ensure_loaded()
        params = params or GenerationParams(
            temperature=self.config.default_temperature,
            top_p=self.config.default_top_p,
        )

        if loaded.backend == "gguf":
            yield from self._stream_gguf(loaded, prompt, params)
            return

        full = self.generate(prompt, params)
        chunk_size = 32
        for i in range(0, len(full), chunk_size):
            yield full[i : i + chunk_size]

    def predict(self, prompt: str, params: GenerationParams | None = None) -> str:
        """Alias para generate()."""
        return self.generate(prompt, params)

    @property
    def metrics(self) -> InferenceMetrics:
        return self._metrics

    def _generate_gguf(self, loaded: LoadedModel, prompt: str, params: GenerationParams) -> str:
        output = loaded.model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop,
        )
        return output["choices"][0]["text"].strip()

    def _stream_gguf(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: GenerationParams,
    ) -> Iterator[str]:
        stream = loaded.model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop,
            stream=True,
        )
        for chunk in stream:
            text = chunk["choices"][0].get("text", "")
            if text:
                yield text

    def _generate_hf(self, loaded: LoadedModel, prompt: str, params: GenerationParams) -> str:
        import torch

        tokenizer = loaded.tokenizer
        model = loaded.model
        if tokenizer is None:
            raise InferenceError("Tokenizer não disponível para backend HF")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=params.temperature > 0,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return text[len(prompt) :].strip()

    def _generate_onnx_stub(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: GenerationParams,
    ) -> str:
        """Stub ONNX — retorna eco formatado até pipeline completo."""
        _ = loaded, params
        return f"[ONNX stub] Resposta para prompt de {len(prompt)} caracteres."