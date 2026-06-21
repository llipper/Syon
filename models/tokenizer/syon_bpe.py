"""
Tokenizer BPE proprietário Syon — treinado do zero no corpus do projeto.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>", "<|code|>", "<|security|>"]


@dataclass
class EncodeResult:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class SyonBPETokenizer:
    """Byte-level BPE simples — sem dependência de modelos pré-treinados."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        max_length: int = 512,
    ) -> None:
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.merges = merges
        self.max_length = max_length
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = vocab[self.pad_token]
        self.bos_token_id = vocab[self.bos_token]
        self.eos_token_id = vocab[self.eos_token]
        self.unk_token_id = vocab[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_pairs(self, word: tuple[str, ...]) -> Counter:
        pairs: Counter = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _merge_word(self, word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        out: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 8192,
        max_length: int = 512,
    ) -> SyonBPETokenizer:
        """Treina BPE do zero em uma lista de textos."""
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        merges: list[tuple[str, str]] = []

        words: Counter = Counter()
        for text in texts:
            for word in re.findall(r"\S+", text.lower()):
                words[tuple(ch for ch in word) + ("</w>",)] += 1

        symbols = set()
        for word in words:
            symbols.update(word)

        next_id = len(vocab)
        for sym in sorted(symbols):
            if sym not in vocab:
                vocab[sym] = next_id
                next_id += 1

        target = max(vocab_size, len(vocab))
        while len(vocab) < target:
            pair_counts: Counter = Counter()
            for word, freq in words.items():
                for pair, c in cls._get_pairs_static(word).items():
                    pair_counts[pair] += c * freq
            if not pair_counts:
                break
            best = pair_counts.most_common(1)[0][0]
            merges.append(best)
            new_token = best[0] + best[1]
            if new_token not in vocab:
                vocab[new_token] = len(vocab)
            new_words: Counter = Counter()
            for word, freq in words.items():
                new_words[cls._merge_word_static(word, best)] = freq
            words = new_words

        return cls(vocab, merges, max_length=max_length)

    @staticmethod
    def _get_pairs_static(word: tuple[str, ...]) -> Counter:
        pairs: Counter = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1
        return pairs

    @staticmethod
    def _merge_word_static(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        out: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(word[i] + word[i + 1])
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def _tokenize_word(self, word: str) -> list[str]:
        chars = tuple(ch for ch in word.lower()) + ("</w>",)
        for pair in self.merges:
            chars = self._merge_word(chars, pair)
        return list(chars)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids: list[int] = [self.bos_token_id] if add_special else []
        for word in re.findall(r"\S+", text):
            for tok in self._tokenize_word(word):
                ids.append(self.vocab.get(tok, self.unk_token_id))
        if add_special:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        words: list[str] = []
        buf = ""
        for tok in tokens:
            if tok in SPECIAL_TOKENS:
                continue
            if tok.endswith("</w>"):
                buf += tok[:-4]
                words.append(buf)
                buf = ""
            else:
                buf += tok
        if buf:
            words.append(buf)
        return " ".join(words)

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = True,
        max_length: int | None = None,
        padding: str = "max_length",
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        max_len = max_length or self.max_length
        ids = self.encode(text)
        if truncation and len(ids) > max_len:
            ids = ids[:max_len]
            ids[-1] = self.eos_token_id

        mask = [1] * len(ids)
        if padding == "max_length":
            pad_len = max_len - len(ids)
            ids.extend([self.pad_token_id] * pad_len)
            mask.extend([0] * pad_len)

        result: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([ids], dtype=torch.long)
            result["attention_mask"] = torch.tensor([mask], dtype=torch.long)
        return result

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "max_length": self.max_length,
            "tokenizer_type": "syon_bpe",
            "from_scratch": True,
        }
        (path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: Path) -> SyonBPETokenizer:
        data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data["merges"]]
        return cls(data["vocab"], merges, max_length=int(data.get("max_length", 512)))

    @classmethod
    def train_from_corpus_dir(cls, data_dir: Path, vocab_size: int = 8192, max_length: int = 512) -> SyonBPETokenizer:
        texts: list[str] = []
        for path in data_dir.rglob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    texts.append(str(obj.get("text", obj.get("content", line))))
                except json.JSONDecodeError:
                    texts.append(line)
        for path in data_dir.rglob("*.txt"):
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        if not texts:
            raise FileNotFoundError(f"Nenhum texto em {data_dir}")
        return cls.train(texts, vocab_size=vocab_size, max_length=max_length)