"""Wrapper de tokenização para inferência Syon."""

from __future__ import annotations

from typing import Any

from syon.exceptions import InferenceError


class Tokenizer:
    """Encapsula tokenizers Hugging Face ou fallback tiktoken."""

    def __init__(self, tokenizer: Any | None = None, model_name: str = "syon-7b"):
        self._hf_tokenizer = tokenizer
        self._tiktoken = None
        self.model_name = model_name

        if tokenizer is None:
            try:
                import tiktoken

                self._tiktoken = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                pass

    @classmethod
    def from_pretrained(cls, model_path: str) -> Tokenizer:
        try:
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained(model_path)
            return cls(tokenizer=hf, model_name=model_path)
        except ImportError as exc:
            raise InferenceError("transformers não instalado para carregar tokenizer") from exc

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        if self._tiktoken is not None:
            return self._tiktoken.encode(text)
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if self._tiktoken is not None:
            return self._tiktoken.decode(token_ids)
        return "".join(chr(t) for t in token_ids if 0 <= t <= 0x10FFFF)

    def encode_special(self, tokens: list[str]) -> list[int]:
        if self._hf_tokenizer is not None and hasattr(self._hf_tokenizer, "convert_tokens_to_ids"):
            return [self._hf_tokenizer.convert_tokens_to_ids(t) for t in tokens]
        return self.encode("".join(tokens))

    def get_token_ids(self, text: str) -> list[int]:
        return self.encode(text)