"""Carregamento de modelos em múltiplos formatos."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from syon.config import SyonConfig
from syon.exceptions import InferenceError, ModelNotFoundError


@dataclass
class LoadedModel:
    """Modelo carregado com metadados de backend."""

    backend: str
    model: Any
    tokenizer: Any | None = None
    model_path: Path = field(default_factory=Path)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelLoader:
    """Carrega modelos Syon em formatos safetensors, GGUF e ONNX."""

    SUPPORTED_EXTENSIONS = {".gguf", ".bin", ".safetensors", ".onnx"}

    def __init__(self, model_path: str | Path, config: SyonConfig | None = None):
        self.model_path = Path(model_path)
        self.config = config or SyonConfig.load()
        self._loaded: LoadedModel | None = None

    def load(self) -> LoadedModel:
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Arquivo de modelo não encontrado: {self.model_path}")

        suffix = self.model_path.suffix.lower()
        if suffix == ".gguf":
            self._loaded = self.load_model_gguf()
        elif suffix in {".bin", ".safetensors"}:
            self._loaded = self.load_model_safetensors()
        elif suffix == ".onnx":
            self._loaded = self.load_model_onnx()
        else:
            if self.model_path.is_dir():
                self._loaded = self.load_model_safetensors()
            else:
                raise ModelNotFoundError(f"Formato não suportado: {suffix}")
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded_model(self) -> LoadedModel | None:
        return self._loaded

    def load_model_gguf(self) -> LoadedModel:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise InferenceError(
                "llama-cpp-python não instalado. Use: pip install llama-cpp-python"
            ) from exc

        inference = self.config.inference_config.get("inference", {})
        llm = Llama(
            model_path=str(self.model_path),
            n_ctx=int(inference.get("n_ctx", self.config.context_window)),
            n_gpu_layers=int(inference.get("n_gpu_layers", -1)),
            verbose=False,
        )
        return LoadedModel(
            backend="gguf",
            model=llm,
            tokenizer=None,
            model_path=self.model_path,
            metadata={"n_ctx": inference.get("n_ctx", self.config.context_window)},
        )

    def load_model_safetensors(self) -> LoadedModel:
        try:
            import json
            import torch
        except ImportError as exc:
            raise InferenceError("torch não instalado") from exc

        path = self.model_path if self.model_path.is_dir() else self.model_path.parent
        cfg_path = path / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("model_type") == "syon_3":
                from models.architecture.syon3 import Syon3
                from models.tokenizer.syon_bpe import SyonBPETokenizer

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = Syon3.from_pretrained(path, device=device)
                tokenizer = SyonBPETokenizer.from_pretrained(path)
                return LoadedModel(
                    backend="syon3",
                    model=model,
                    tokenizer=tokenizer,
                    model_path=path,
                    metadata={"model_type": "syon_3", "parameters": model.num_parameters()},
                )

        raise InferenceError(
            f"Modelo não reconhecido em {path}. Use diretório Syon 3 (config.json com model_type=syon_3)."
        )

    def load_model_onnx(self) -> LoadedModel:
        """Stub para carregamento ONNX — requer onnxruntime em produção."""
        if not self.model_path.exists():
            raise ModelNotFoundError(f"Modelo ONNX não encontrado: {self.model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise InferenceError(
                "onnxruntime não instalado. Use: pip install onnxruntime"
            ) from exc

        session = ort.InferenceSession(str(self.model_path))
        return LoadedModel(
            backend="onnx",
            model=session,
            tokenizer=None,
            model_path=self.model_path,
            metadata={"providers": session.get_providers()},
        )

    def load_quantized_model(self, quantization: str = "q4_0") -> LoadedModel:
        """Carrega variante GGUF quantizada a partir do diretório do modelo."""
        if self.model_path.suffix.lower() == ".gguf":
            return self.load_model_gguf()

        gguf_dir = self.model_path.parent / f"{self.model_path.stem}-gguf" / quantization
        candidates = list(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
        if not candidates:
            raise ModelNotFoundError(
                f"Modelo quantizado não encontrado em {gguf_dir} (quantization={quantization})"
            )
        original_path = self.model_path
        self.model_path = candidates[0]
        try:
            return self.load_model_gguf()
        finally:
            self.model_path = original_path