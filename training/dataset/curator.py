"""Curadoria independente de dados para treinamento base."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from training.dataset.composition import DatasetComposition, load_composition


@dataclass
class CuratedSample:
    text: str
    domain: str
    source: str
    metadata: dict


class DatasetCurator:
    """Pipeline de cura, deduplicação e validação de amostras."""

    def __init__(self, data_dir: Path, composition: DatasetComposition | None = None):
        self.data_dir = Path(data_dir)
        self.composition = composition or load_composition()
        self._seen_hashes: set[str] = set()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def deduplicate(self, text: str) -> bool:
        digest = self._hash(text)
        if digest in self._seen_hashes:
            return False
        self._seen_hashes.add(digest)
        return True

    @staticmethod
    def _turns_to_syon_chat(turns: list[dict]) -> str:
        role_map = {"user": "<|user|>", "model": "<|assistant|>", "assistant": "<|assistant|>"}
        parts: list[str] = []
        for t in turns:
            role = str(t.get("role", "")).lower()
            tag = role_map.get(role)
            content = str(t.get("content", "")).strip()
            if tag and content:
                parts.append(f"{tag}{content}")
        return "".join(parts)

    def load_jsonl(self, path: Path, domain: str, source: str) -> Iterator[CuratedSample]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = record.get("text", "")
                if not text and isinstance(record.get("turns"), list):
                    text = self._turns_to_syon_chat(record["turns"])
                if not text or not self.deduplicate(text):
                    continue
                yield CuratedSample(
                    text=text,
                    domain=domain,
                    source=source,
                    metadata=record.get("metadata", {}),
                )

    def curate_domain(self, domain: str, sources: list[str]) -> list[CuratedSample]:
        samples: list[CuratedSample] = []
        domain_dir = self.data_dir / domain
        if not domain_dir.exists():
            return samples
        for source in sources:
            path = domain_dir / f"{source}.jsonl"
            if path.exists():
                samples.extend(self.load_jsonl(path, domain=domain, source=source))
        return samples

    def curate_all(self) -> list[CuratedSample]:
        all_samples: list[CuratedSample] = []
        for slice_ in self.composition.slices:
            all_samples.extend(self.curate_domain(slice_.name, slice_.sources))
        return all_samples