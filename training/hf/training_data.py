"""
Dados de treino Syon 3 — conversação + raciocínio.

Fontes: arquivos locais (prioridade) ou API datasets do Hugging Face (só texto).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from models.tokenizer.syon_bpe import SyonBPETokenizer
from training.dataset.composition import load_composition
from training.dataset.curator import DatasetCurator
from training.hf.conversation_data import (
    build_hf_dataset,
    load_conversation_records,
    normalize_turns,
)
from training.hf.policy import assert_dataset_allowed


def format_reasoning_sample(text: str, domain: str, source: str) -> dict[str, Any] | None:
    text = text.strip()
    if len(text) < 80:
        return None
    return {
        "turns": [
            {
                "role": "user",
                "content": (
                    f"Analise o tema ({domain}/{source}) e responda com "
                    "raciocínio passo a passo, claro e completo."
                ),
            },
            {"role": "assistant", "content": text[:3000]},
        ]
    }


def load_reasoning_records(
    raw_dir: Path,
    *,
    domains: list[str],
    max_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if not raw_dir.exists():
        print(f"[Syon/HF] Sem dados de raciocínio em {raw_dir}")
        return []

    curator = DatasetCurator(raw_dir, load_composition())
    records: list[dict[str, Any]] = []

    slice_map = {s.name: s.sources for s in curator.composition.slices}
    for domain in domains:
        sources = slice_map.get(domain, [])
        if not sources:
            continue
        for sample in curator.curate_domain(domain, sources):
            item = format_reasoning_sample(sample.text, sample.domain, sample.source)
            if item:
                records.append(item)

    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:max_samples]
    print(f"[Syon/HF] {len(records)} amostras de raciocínio ({domains})")
    return records


def load_syon3_training_records(
    *,
    conversation_dir: Path,
    raw_dir: Path,
    max_conversation: int,
    max_reasoning: int,
    reasoning_domains: list[str],
    hf_dataset: str | None,
    hf_split: str,
    conversation_weight: float = 0.6,
) -> list[dict[str, Any]]:
    local_files = [
        conversation_dir / "instruct_aira_pt.jsonl",
        conversation_dir / "ultrachat_br.jsonl",
    ]
    conv = load_conversation_records(
        local_files=local_files,
        hf_dataset=None,
        hf_split=hf_split,
        max_samples=max_conversation,
    )

    if len(conv) < max_conversation and hf_dataset:
        assert_dataset_allowed(hf_dataset)
        print(f"[Syon/HF] API datasets — baixando texto: {hf_dataset}")
        need = max_conversation - len(conv)
        extra = load_conversation_records(
            local_files=[],
            hf_dataset=hf_dataset,
            hf_split=hf_split,
            max_samples=need,
        )
        conv.extend(extra)

    reasoning = load_reasoning_records(
        raw_dir,
        domains=reasoning_domains,
        max_samples=max_reasoning,
    )

    if not conv and not reasoning:
        raise ValueError("Nenhum dado de treino. Importe conversação ou gere curriculum.")

    if conv and reasoning:
        n_conv = int((max_conversation + max_reasoning) * conversation_weight)
        n_conv = min(n_conv, len(conv))
        n_reas = min(max_reasoning, len(reasoning))
        rng = random.Random(42)
        rng.shuffle(conv)
        rng.shuffle(reasoning)
        mixed = conv[:n_conv] + reasoning[:n_reas]
        rng.shuffle(mixed)
        print(f"[Syon/HF] Mix: {n_conv} conversação + {n_reas} raciocínio = {len(mixed)}")
        return mixed

    return conv or reasoning


def load_syon3_hf_dataset(
    tokenizer: SyonBPETokenizer,
    *,
    conversation_dir: Path,
    raw_dir: Path,
    max_length: int,
    max_conversation: int,
    max_reasoning: int,
    reasoning_domains: list[str],
    hf_dataset: str | None,
    hf_split: str = "portuguese",
    conversation_weight: float = 0.6,
    val_ratio: float = 0.02,
) -> tuple[Dataset, Dataset | None]:
    records = load_syon3_training_records(
        conversation_dir=conversation_dir,
        raw_dir=raw_dir,
        max_conversation=max_conversation,
        max_reasoning=max_reasoning,
        reasoning_domains=reasoning_domains,
        hf_dataset=hf_dataset,
        hf_split=hf_split,
        conversation_weight=conversation_weight,
    )

    dataset = build_hf_dataset(tokenizer, records, max_length)
    if val_ratio <= 0 or len(dataset) < 50:
        return dataset, None
    split = dataset.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]